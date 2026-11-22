# AUDIT COPY of the experiment code found in saved notebooks / scripts. It is READ-ONLY evidence, not meant to be run as one file.
# Notebooks are the SAVED versions: a cell may have been edited after it was executed. 'exec' is the cell's execution counter if the notebook stored one.


# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 0 | exec 1
# ====================================================================================================
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
!ls /content/gdrive/MyDrive
!ls /content/gdrive/MyDrive/BASIL_datasets/corrected

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 1 | exec None
# ====================================================================================================
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 2 | exec None
# ====================================================================================================
!pip -q install datasets
from datasets import load_dataset
ds = load_dataset("mediabiasgroup/anno-lexical")
print(ds)
print(ds["train"].column_names)
print(ds["train"][0])

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 3 | exec None
# ====================================================================================================
# ============================================================
# STAGED TRAINING:  Anno-lexical (stage 1)  ->  corrected BASIL (stage 2)
# Same model and BASIL settings as the baseline (xlm-roberta-base, sampling 5 neutral per biased, 8 epochs, lr 1e-5)
# so the result can be compared fairly with runs/xlm-roberta-base_sample5.
# Stage 1 is trained ONCE (seed 42) and saved to Drive. Stage 2 runs for seeds 42, 43, 44, each starting from that checkpoint.
# BASIL validation + test are never used in stage 1. Epoch/threshold chosen on BASIL VALIDATION only; test scored once.
# ============================================================
import os, random
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix, accuracy_score, f1_score)

# ---------------- where the files are ----------------
USE_DRIVE   = True
DRIVE_MOUNT = "/content/gdrive"          # the mount that worked for you

# ---------------- settings ----------------
MODEL_NAME  = "xlm-roberta-base"         # SAME as the baseline
SEEDS       = [42, 43, 44]
# stage 1 (Anno-lexical)
S1_EPOCHS, S1_BATCH, S1_LR, S1_MAX_LEN = 2, 32, 2e-5, 128
S1_MAX_TRAIN = None                      # None = all 33,831 sentences. e.g. 15000 to run faster if GPU time is short
# stage 2 (BASIL) - identical to the baseline
EPOCHS, BATCH, LR, MAX_LEN = 8, 16, 1e-5, 256
NEG_PER_POS = 5
RUN_NAME    = "xlm-roberta-base_anno_then_basil"
OVERWRITE   = False                      # False = finished stage-1 checkpoint / (seed) runs are reused from Drive

if USE_DRIVE:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT)
    mydrive = f"{DRIVE_MOUNT}/MyDrive"
    BASE = f"{mydrive}/BASIL_datasets"; DATA = f"{BASE}/corrected"; OUT = f"{BASE}/runs/{RUN_NAME}"
else:
    BASE = DATA = "/content"; OUT = f"/content/runs/{RUN_NAME}"
BASELINE_RESULTS = f"{BASE}/runs/xlm-roberta-base_sample5/results.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| run:", RUN_NAME)
if device == "cpu": raise SystemExit("No GPU: connect to a GPU runtime (or use Kaggle) before running this cell.")

# ---- HELPERS (same as the baseline cell) ----
def pick_threshold(y, p):
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.linspace(0.02, 0.98, 97), 2):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1 + 1e-12: best_t, best_f1 = float(t), f1
    return best_t, best_f1

def metrics(y, p, t):
    pred = (p >= t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=t, precision=pr[0], recall=rc[0], f1=f1[0],
                macro_f1=f1_score(y, pred, average="macro", zero_division=0), accuracy=accuracy_score(y, pred),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), predicted_biased=int(pred.sum()))

def bootstrap_by_story(y, p, t, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed); g = pd.Series(groups).values
    idx_of = {s: np.where(g == s)[0] for s in np.unique(g)}; keys = list(idx_of)
    f1s, aps = [], []
    for _ in range(n):
        pick = np.concatenate([idx_of[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        if y[pick].sum() == 0: continue
        f1s.append(f1_score(y[pick], (p[pick] >= t).astype(int), zero_division=0)); aps.append(average_precision_score(y[pick], p[pick]))
    return np.percentile(f1s, [2.5, 97.5]), np.percentile(aps, [2.5, 97.5])

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
collate = DataCollatorWithPadding(tok)

class DS(Dataset):
    def __init__(self, enc, labels): self.enc, self.labels, self.keys = enc, labels, [k for k in enc.keys() if k in ("input_ids", "attention_mask", "token_type_ids")]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.keys}; item["labels"] = int(self.labels[i]); return item

@torch.no_grad()
def predict(model, ds):
    model.eval(); out = []
    for b in DataLoader(ds, batch_size=64, collate_fn=collate):
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def new_scaler():
    try:    return torch.amp.GradScaler("cuda")
    except Exception: return torch.cuda.amp.GradScaler()

# ============================================================
# STAGE 1: Anno-lexical
# ============================================================
S1_DIR = f"{OUT}/stage1_anno_checkpoint"
os.makedirs(OUT, exist_ok=True)
if os.path.exists(f"{S1_DIR}/config.json") and not OVERWRITE:
    print("Stage 1 checkpoint found on Drive -> skipping stage 1:", S1_DIR)
else:
    from datasets import load_dataset
    an = load_dataset("mediabiasgroup/anno-lexical")
    a_tr, a_va, a_te = [an[s].to_pandas() for s in ["train", "validation", "test"]]
    if S1_MAX_TRAIN: a_tr = a_tr.sample(S1_MAX_TRAIN, random_state=0).reset_index(drop=True)
    print(f"Anno-lexical: train {len(a_tr)} ({a_tr.label.mean():.1%} biased) | dev {len(a_va)} | test {len(a_te)}")
    enc = lambda d: tok(list(d.text), truncation=True, max_length=S1_MAX_LEN)
    ds1_tr, ds1_va, ds1_te = DS(enc(a_tr), a_tr.label.values), DS(enc(a_va), a_va.label.values), DS(enc(a_te), a_te.label.values)
    set_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    steps = int(np.ceil(len(ds1_tr) / S1_BATCH)) * S1_EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=S1_LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); scaler = new_scaler()
    best_auc, best_state = -1, None
    for ep in range(S1_EPOCHS):
        model.train(); total, n = 0.0, 0
        for b in DataLoader(ds1_tr, batch_size=S1_BATCH, shuffle=True, collate_fn=collate):
            b = {k: v.to(device) for k, v in b.items()}; labels = b.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**b).logits
            loss = F.cross_entropy(logits.float(), labels)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            total += loss.item(); n += 1
        auc = roc_auc_score(a_va.label.values, predict(model, ds1_va))
        print(f"   [stage 1] epoch {ep+1}/{S1_EPOCHS} | train loss {total/n:.3f} | Anno-lexical dev ROC-AUC {auc:.3f}")
        if auc > best_auc: best_auc, best_state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"   [stage 1] Anno-lexical TEST ROC-AUC (sanity check on its own test): {roc_auc_score(a_te.label.values, predict(model, ds1_te)):.3f}")
    model.save_pretrained(S1_DIR); print("Saved stage-1 checkpoint ->", S1_DIR)
    del model, opt; torch.cuda.empty_cache()

# ============================================================
# STAGE 2: corrected BASIL, starting from the stage-1 checkpoint
# ============================================================
need = [f"{DATA}/{n}_corrected.csv" for n in ["train", "validation", "test"]]
if any(not os.path.exists(p) for p in need): raise FileNotFoundError(f"Missing BASIL files: {need}")
tr, va, te = [pd.read_csv(f"{DATA}/{n}_corrected.csv") for n in ["train", "validation", "test"]]
for d in (tr, va, te): d["target_sentence"] = d["target_sentence"].fillna("")
y_tr, y_va, y_te = tr.label.values, va.label.values, te.label.values
print(f"BASIL: train {len(tr)} ({y_tr.sum()} biased) | validation {len(va)} ({y_va.sum()}) | test {len(te)} ({y_te.sum()})")
enc2 = lambda d: tok(list(d.target_sentence), truncation=True, max_length=MAX_LEN)
ds_tr, ds_va, ds_te = DS(enc2(tr), y_tr), DS(enc2(va), y_va), DS(enc2(te), y_te)

def train_stage2(seed):
    set_seed(seed); rng = np.random.default_rng(seed)
    model = AutoModelForSequenceClassification.from_pretrained(S1_DIR, num_labels=2).to(device)
    pos_idx, neg_idx = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
    def epoch_idx():
        idx = np.concatenate([pos_idx, rng.choice(neg_idx, size=min(len(neg_idx), NEG_PER_POS * len(pos_idx)), replace=False)])
        rng.shuffle(idx); return idx
    steps_per_epoch = int(np.ceil(len(epoch_idx()) / BATCH))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps_per_epoch * EPOCHS), steps_per_epoch * EPOCHS); scaler = new_scaler()
    best_ap, best_state = -1, None
    for ep in range(EPOCHS):
        model.train(); total = 0.0
        for b in DataLoader(Subset(ds_tr, epoch_idx()), batch_size=BATCH, collate_fn=collate):
            b = {k: v.to(device) for k, v in b.items()}; labels = b.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**b).logits
            loss = F.cross_entropy(logits.float(), labels)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); total += loss.item()
        ap = average_precision_score(y_va, predict(model, ds_va))
        print(f"   [stage 2 seed {seed}] epoch {ep+1}/{EPOCHS} | train loss {total/steps_per_epoch:.3f} | BASIL validation PR-AUC {ap:.3f}")
        if ap > best_ap: best_ap, best_state = ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pv, pt = predict(model, ds_va), predict(model, ds_te)
    del model, opt; torch.cuda.empty_cache()
    return pv, pt

rows, pvs, pts = [], [], []
for seed in SEEDS:
    f = f"{OUT}/anno_then_basil_seed{seed}_probs.npz"
    if os.path.exists(f) and not OVERWRITE:
        z = np.load(f); pv, pt = z["val"], z["test"]; print(f"[seed {seed}] loaded from disk")
    else:
        pv, pt = train_stage2(seed); np.savez(f, val=pv, test=pt)
    t, _ = pick_threshold(y_va, pv); rows.append(dict(mode="anno_then_basil", seed=seed, **metrics(y_te, pt, t))); pvs.append(pv); pts.append(pt)
pv, pt = np.mean(pvs, 0), np.mean(pts, 0)
t, _ = pick_threshold(y_va, pv); m = metrics(y_te, pt, t)
f1_ci, ap_ci = bootstrap_by_story(y_te, pt, t, te.story_id.values)
rows.append(dict(mode="anno_then_basil", seed="ensemble", **m, f1_ci=f"{f1_ci[0]:.3f}-{f1_ci[1]:.3f}", pr_auc_ci=f"{ap_ci[0]:.3f}-{ap_ci[1]:.3f}"))
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/results.csv", index=False)
pd.DataFrame({"sentenceId": te.sentenceId, "label": y_te, "prob": pt, "pred": (pt >= t).astype(int)}).to_csv(f"{OUT}/ensemble_test_predictions.csv", index=False)

show = ["mode", "seed", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "fp", "fn"]
print("\n=== Anno-lexical -> BASIL (test set; threshold chosen on validation) ==="); print(res[show].round(3).to_string(index=False))
r = res[res.seed != "ensemble"]
print("\nmean +/- std over seeds:"); print(r[["precision", "recall", "f1", "pr_auc", "roc_auc"]].agg(["mean", "std"]).round(3).to_string())
print("\nensemble 95% intervals (resampling stories):"); print(res[res.seed == "ensemble"][["f1", "f1_ci", "pr_auc", "pr_auc_ci"]].round(3).to_string(index=False))
if os.path.exists(BASELINE_RESULTS):
    b = pd.read_csv(BASELINE_RESULTS); b = b[(b["mode"] == "target") & (b.seed.astype(str) != "ensemble")]
    print("\n=== BASELINE (BASIL only, same settings), mean over seeds ==="); print(b[["precision", "recall", "f1", "pr_auc", "roc_auc"]].mean().round(3).to_string())
print("\nSaved to:", OUT)

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 4 | exec None
# ====================================================================================================
from google.colab import files
import os, shutil
os.makedirs("/content/gdrive/MyDrive/BASIL_datasets/babe", exist_ok=True)
up = files.upload()
for n in up: shutil.move(n, f"/content/gdrive/MyDrive/BASIL_datasets/babe/{n}")
print(os.listdir("/content/gdrive/MyDrive/BASIL_datasets/babe"))

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 5 | exec None
# ====================================================================================================
# ============================================================
# STAGE 1.5 = BABE, then STAGE 2 = corrected BASIL
#   START = "anno" : Anno-lexical (saved checkpoint) -> BABE -> BASIL      (main result)
#   START = "base" : plain xlm-roberta-base          -> BABE -> BASIL      (ablation: BABE alone)
# Same BASIL settings as the baseline and the Anno-lexical run, so the three can be compared fairly.
# BASIL validation + test are never used before the final stage. Epoch/threshold chosen on BASIL VALIDATION only.
# ============================================================
import os, re, ast, random
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix, accuracy_score, f1_score)

# ---------------- settings ----------------
START       = "anno"                     # "anno" (main) or "base" (ablation)
DRIVE_MOUNT = "/content/gdrive"
MODEL_NAME  = "xlm-roberta-base"
SEEDS       = [42, 43, 44]
# BABE stage
B_EPOCHS, B_BATCH, B_LR, B_MAX_LEN = 3, 16, 1e-5, 128
MIN_COUNT, RATE_MIN = 8, 0.30            # same BABE word filter as before
MAX_PER_OUTLET = None                    # e.g. 400 to cap each outlet if you want to reduce outlet skew
# BASIL stage (identical to the baseline)
EPOCHS, BATCH, LR, MAX_LEN = 8, 16, 1e-5, 256
NEG_PER_POS = 5
OVERWRITE   = False
RUN_NAME    = "xlm-roberta-base_" + ("anno_babe_basil" if START == "anno" else "babe_basil")

from google.colab import drive
drive.mount(DRIVE_MOUNT)
BASE = f"{DRIVE_MOUNT}/MyDrive/BASIL_datasets"; DATA = f"{BASE}/corrected"; OUT = f"{BASE}/runs/{RUN_NAME}"
BABE_PATH = f"{BASE}/babe/final_labels_SG2.xlsx"
S1_DIR = f"{BASE}/runs/xlm-roberta-base_anno_then_basil/stage1_anno_checkpoint"
INIT = S1_DIR if START == "anno" else MODEL_NAME
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| run:", RUN_NAME, "| start from:", INIT)
if device == "cpu": raise SystemExit("No GPU: connect to a GPU runtime first.")
if START == "anno" and not os.path.exists(f"{S1_DIR}/config.json"): raise FileNotFoundError(f"Stage-1 checkpoint not found: {S1_DIR}. Run the Anno-lexical cell first.")
os.makedirs(OUT, exist_ok=True)

# ---------------- helpers ----------------
def pick_threshold(y, p):
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.linspace(0.02, 0.98, 97), 2):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1 + 1e-12: best_t, best_f1 = float(t), f1
    return best_t, best_f1

def metrics(y, p, t):
    pred = (p >= t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=t, precision=pr[0], recall=rc[0], f1=f1[0],
                macro_f1=f1_score(y, pred, average="macro", zero_division=0), accuracy=accuracy_score(y, pred),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), predicted_biased=int(pred.sum()))

def bootstrap_by_story(y, p, t, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed); g = pd.Series(groups).values
    idx_of = {s: np.where(g == s)[0] for s in np.unique(g)}; keys = list(idx_of)
    f1s, aps = [], []
    for _ in range(n):
        pick = np.concatenate([idx_of[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        if y[pick].sum() == 0: continue
        f1s.append(f1_score(y[pick], (p[pick] >= t).astype(int), zero_division=0)); aps.append(average_precision_score(y[pick], p[pick]))
    return np.percentile(f1s, [2.5, 97.5]), np.percentile(aps, [2.5, 97.5])

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
collate = DataCollatorWithPadding(tok)

class DS(Dataset):
    def __init__(self, enc, labels): self.enc, self.labels, self.keys = enc, labels, [k for k in enc.keys() if k in ("input_ids", "attention_mask", "token_type_ids")]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.keys}; item["labels"] = int(self.labels[i]); return item

@torch.no_grad()
def predict(model, ds):
    model.eval(); out = []
    for b in DataLoader(ds, batch_size=64, collate_fn=collate):
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def new_scaler():
    try:    return torch.amp.GradScaler("cuda")
    except Exception: return torch.cuda.amp.GradScaler()

# ---------------- load BASIL (used for the overlap check and stage 2) ----------------
need = [f"{DATA}/{n}_corrected.csv" for n in ["train", "validation", "test"]]
if any(not os.path.exists(p) for p in need): raise FileNotFoundError(f"Missing BASIL files: {need}")
tr, va, te = [pd.read_csv(f"{DATA}/{n}_corrected.csv") for n in ["train", "validation", "test"]]
for d in (tr, va, te): d["target_sentence"] = d["target_sentence"].fillna("")
y_tr, y_va, y_te = tr.label.values, va.label.values, te.label.values
print(f"BASIL: train {len(tr)} ({y_tr.sum()} biased) | validation {len(va)} ({y_va.sum()}) | test {len(te)} ({y_te.sum()})")

# ============================================================
# BABE: select sentences whose bias is carried by loaded words (same rule as before)
# ============================================================
if not os.path.exists(BABE_PATH): raise FileNotFoundError(f"Upload final_labels_SG2.xlsx to {BABE_PATH}")
df = pd.read_excel(BABE_PATH)
df = df[df["label_bias"].isin(["Biased", "Non-biased"])].copy()
def parse(c):
    if isinstance(c, str) and c.strip().startswith("["):
        try: return [str(w).strip().lower() for w in ast.literal_eval(c)]
        except Exception: return []
    return []
df["marked"] = df["biased_words"].apply(parse)
df["toks"]   = df["text"].apply(lambda s: re.findall(r"[a-z][a-z'\-]+", str(s).lower()))
STOP = set("""a an the and or but of to in on at by for with from up out down off over as is are was were be been being
it its this that these those he she they them his her their we you i not no nor so than then there here who whom which what
when where how all any some more most other such only own same too very can will just do does did has have had said says say
would could should may might about into after before between through during under again further once""".split())
appear, marked = Counter(), Counter()
for toks, mk, lab in zip(df.toks, df.marked, df.label_bias):
    for w in set(toks): appear[w] += 1
    if lab == "Biased":
        for w in set(mk): marked[w] += 1
def is_cue(w):
    if w in STOP or len(w) < 3: return False
    if appear[w] >= MIN_COUNT: return marked[w] / appear[w] >= RATE_MIN
    return True
df["cue_words"] = df.marked.apply(lambda ws: [w for w in ws if is_cue(w)])
df["n_cues"] = df.cue_words.str.len()
pos = df[(df.label_bias == "Biased") & (df.n_cues >= 1)].copy()
neg = df[df.label_bias == "Non-biased"].copy()
if MAX_PER_OUTLET: pos = pos.sort_values("n_cues", ascending=False).groupby("outlet", group_keys=False).head(MAX_PER_OUTLET)
babe = pd.concat([pos.assign(label=1), neg.assign(label=0)], ignore_index=True).rename(columns={"text": "target_sentence"})
babe = babe[["target_sentence", "label", "outlet"]].drop_duplicates("target_sentence").reset_index(drop=True)

# leakage check: BABE sentences identical to BASIL validation/test sentences are removed
norm = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()
held = set(map(norm, pd.concat([va.target_sentence, te.target_sentence])))
dup = babe.target_sentence.map(lambda s: norm(s) in held)
print(f"BABE sentences identical to BASIL validation/test sentences: {int(dup.sum())} (removed)")
babe = babe[~dup].reset_index(drop=True)
print(f"BABE selected: {len(babe)} sentences | lexical (1): {int(babe.label.sum())} | non-biased (0): {int((babe.label==0).sum())}")
print("outlet share among selected positives:", babe[babe.label == 1].outlet.value_counts(normalize=True).round(2).head(4).to_dict())

# ============================================================
# BABE fine-tuning (10% of BABE held out only to pick the best epoch)
# ============================================================
B_DIR = f"{OUT}/babe_checkpoint"
if os.path.exists(f"{B_DIR}/config.json") and not OVERWRITE:
    print("BABE-stage checkpoint found on Drive -> skipping:", B_DIR)
else:
    b_tr, b_va = train_test_split(babe, test_size=0.1, stratify=babe.label, random_state=0)
    enc = lambda d: tok(list(d.target_sentence), truncation=True, max_length=B_MAX_LEN)
    ds_btr, ds_bva = DS(enc(b_tr), b_tr.label.values), DS(enc(b_va), b_va.label.values)
    set_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(INIT, num_labels=2).to(device)
    steps = int(np.ceil(len(ds_btr) / B_BATCH)) * B_EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=B_LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); scaler = new_scaler()
    best_auc, best_state = -1, None
    for ep in range(B_EPOCHS):
        model.train(); total, n = 0.0, 0
        for b in DataLoader(ds_btr, batch_size=B_BATCH, shuffle=True, collate_fn=collate):
            b = {k: v.to(device) for k, v in b.items()}; labels = b.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**b).logits
            loss = F.cross_entropy(logits.float(), labels)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            total += loss.item(); n += 1
        auc = roc_auc_score(b_va.label.values, predict(model, ds_bva))
        print(f"   [BABE stage] epoch {ep+1}/{B_EPOCHS} | train loss {total/n:.3f} | held-out BABE ROC-AUC {auc:.3f}")
        if auc > best_auc: best_auc, best_state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.save_pretrained(B_DIR); print("Saved BABE-stage checkpoint ->", B_DIR)
    del model, opt; torch.cuda.empty_cache()

# ============================================================
# STAGE 2: corrected BASIL, starting from the BABE-stage checkpoint
# ============================================================
enc2 = lambda d: tok(list(d.target_sentence), truncation=True, max_length=MAX_LEN)
ds_tr, ds_va, ds_te = DS(enc2(tr), y_tr), DS(enc2(va), y_va), DS(enc2(te), y_te)

def train_stage2(seed):
    set_seed(seed); rng = np.random.default_rng(seed)
    model = AutoModelForSequenceClassification.from_pretrained(B_DIR, num_labels=2).to(device)
    pos_idx, neg_idx = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
    def epoch_idx():
        idx = np.concatenate([pos_idx, rng.choice(neg_idx, size=min(len(neg_idx), NEG_PER_POS * len(pos_idx)), replace=False)])
        rng.shuffle(idx); return idx
    steps_per_epoch = int(np.ceil(len(epoch_idx()) / BATCH))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps_per_epoch * EPOCHS), steps_per_epoch * EPOCHS); scaler = new_scaler()
    best_ap, best_state = -1, None
    for ep in range(EPOCHS):
        model.train(); total = 0.0
        for b in DataLoader(Subset(ds_tr, epoch_idx()), batch_size=BATCH, collate_fn=collate):
            b = {k: v.to(device) for k, v in b.items()}; labels = b.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**b).logits
            loss = F.cross_entropy(logits.float(), labels)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); total += loss.item()
        ap = average_precision_score(y_va, predict(model, ds_va))
        print(f"   [BASIL seed {seed}] epoch {ep+1}/{EPOCHS} | train loss {total/steps_per_epoch:.3f} | BASIL validation PR-AUC {ap:.3f}")
        if ap > best_ap: best_ap, best_state = ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pv, pt = predict(model, ds_va), predict(model, ds_te)
    del model, opt; torch.cuda.empty_cache()
    return pv, pt

mode = RUN_NAME.replace("xlm-roberta-base_", "")
rows, pvs, pts = [], [], []
for seed in SEEDS:
    f = f"{OUT}/{mode}_seed{seed}_probs.npz"
    if os.path.exists(f) and not OVERWRITE:
        z = np.load(f); pv, pt = z["val"], z["test"]; print(f"[seed {seed}] loaded from disk")
    else:
        pv, pt = train_stage2(seed); np.savez(f, val=pv, test=pt)
    t, _ = pick_threshold(y_va, pv); rows.append(dict(mode=mode, seed=seed, **metrics(y_te, pt, t))); pvs.append(pv); pts.append(pt)
pv, pt = np.mean(pvs, 0), np.mean(pts, 0)
t, _ = pick_threshold(y_va, pv); m = metrics(y_te, pt, t)
f1_ci, ap_ci = bootstrap_by_story(y_te, pt, t, te.story_id.values)
rows.append(dict(mode=mode, seed="ensemble", **m, f1_ci=f"{f1_ci[0]:.3f}-{f1_ci[1]:.3f}", pr_auc_ci=f"{ap_ci[0]:.3f}-{ap_ci[1]:.3f}"))
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/results.csv", index=False)
pd.DataFrame({"sentenceId": te.sentenceId, "label": y_te, "prob": pt, "pred": (pt >= t).astype(int)}).to_csv(f"{OUT}/ensemble_test_predictions.csv", index=False)

show = ["mode", "seed", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "fp", "fn"]
print(f"\n=== {mode} (test set; threshold chosen on validation) ==="); print(res[show].round(3).to_string(index=False))
print("\nensemble 95% intervals (resampling stories):"); print(res[res.seed == "ensemble"][["f1", "f1_ci", "pr_auc", "pr_auc_ci"]].round(3).to_string(index=False))

# ---- side-by-side with the earlier runs (mean over seeds, test set) ----
comp = {}
for label, folder in [("BASIL only", "xlm-roberta-base_sample5"), ("Anno -> BASIL", "xlm-roberta-base_anno_then_basil"), (mode.replace("_", " -> ").upper().replace("->", "->"), RUN_NAME)]:
    p = f"{BASE}/runs/{folder}/results.csv"
    if os.path.exists(p):
        r = pd.read_csv(p); r = r[r.seed.astype(str) != "ensemble"]
        if "mode" in r and folder == "xlm-roberta-base_sample5": r = r[r["mode"] == "target"]
        comp[label] = r[["precision", "recall", "f1", "pr_auc", "roc_auc"]].astype(float).mean()
print("\n=== comparison (mean over seeds, BASIL test set) ==="); print(pd.DataFrame(comp).round(3).to_string())
print("\nSaved to:", OUT)

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 6 | exec None
# ====================================================================================================
# ============================================================
# STAGE 1.5 = BABE, then STAGE 2 = corrected BASIL
#   START = "anno" : Anno-lexical (saved checkpoint) -> BABE -> BASIL      (main result)
#   START = "base" : plain xlm-roberta-base          -> BABE -> BASIL      (ablation: BABE alone)
# Same BASIL settings as the baseline and the Anno-lexical run, so the three can be compared fairly.
# BASIL validation + test are never used before the final stage. Epoch/threshold chosen on BASIL VALIDATION only.
# ============================================================
import os, re, ast, random
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix, accuracy_score, f1_score)

# ---------------- settings ----------------
START       = "base"                     # "anno" (main) or "base" (ablation)
DRIVE_MOUNT = "/content/gdrive"
MODEL_NAME  = "xlm-roberta-base"
SEEDS       = [42, 43, 44]
# BABE stage
B_EPOCHS, B_BATCH, B_LR, B_MAX_LEN = 3, 16, 1e-5, 128
MIN_COUNT, RATE_MIN = 8, 0.30            # same BABE word filter as before
MAX_PER_OUTLET = None                    # e.g. 400 to cap each outlet if you want to reduce outlet skew
# BASIL stage (identical to the baseline)
EPOCHS, BATCH, LR, MAX_LEN = 8, 16, 1e-5, 256
NEG_PER_POS = 5
OVERWRITE   = False
RUN_NAME    = "xlm-roberta-base_" + ("anno_babe_basil" if START == "anno" else "babe_basil")

from google.colab import drive
drive.mount(DRIVE_MOUNT)
BASE = f"{DRIVE_MOUNT}/MyDrive/BASIL_datasets"; DATA = f"{BASE}/corrected"; OUT = f"{BASE}/runs/{RUN_NAME}"
BABE_PATH = f"{BASE}/babe/final_labels_SG2.xlsx"
S1_DIR = f"{BASE}/runs/xlm-roberta-base_anno_then_basil/stage1_anno_checkpoint"
INIT = S1_DIR if START == "anno" else MODEL_NAME
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| run:", RUN_NAME, "| start from:", INIT)
if device == "cpu": raise SystemExit("No GPU: connect to a GPU runtime first.")
if START == "anno" and not os.path.exists(f"{S1_DIR}/config.json"): raise FileNotFoundError(f"Stage-1 checkpoint not found: {S1_DIR}. Run the Anno-lexical cell first.")
os.makedirs(OUT, exist_ok=True)

# ---------------- helpers ----------------
def pick_threshold(y, p):
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.linspace(0.02, 0.98, 97), 2):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1 + 1e-12: best_t, best_f1 = float(t), f1
    return best_t, best_f1

def metrics(y, p, t):
    pred = (p >= t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=t, precision=pr[0], recall=rc[0], f1=f1[0],
                macro_f1=f1_score(y, pred, average="macro", zero_division=0), accuracy=accuracy_score(y, pred),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), predicted_biased=int(pred.sum()))

def bootstrap_by_story(y, p, t, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed); g = pd.Series(groups).values
    idx_of = {s: np.where(g == s)[0] for s in np.unique(g)}; keys = list(idx_of)
    f1s, aps = [], []
    for _ in range(n):
        pick = np.concatenate([idx_of[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        if y[pick].sum() == 0: continue
        f1s.append(f1_score(y[pick], (p[pick] >= t).astype(int), zero_division=0)); aps.append(average_precision_score(y[pick], p[pick]))
    return np.percentile(f1s, [2.5, 97.5]), np.percentile(aps, [2.5, 97.5])

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
collate = DataCollatorWithPadding(tok)

class DS(Dataset):
    def __init__(self, enc, labels): self.enc, self.labels, self.keys = enc, labels, [k for k in enc.keys() if k in ("input_ids", "attention_mask", "token_type_ids")]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.keys}; item["labels"] = int(self.labels[i]); return item

@torch.no_grad()
def predict(model, ds):
    model.eval(); out = []
    for b in DataLoader(ds, batch_size=64, collate_fn=collate):
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def new_scaler():
    try:    return torch.amp.GradScaler("cuda")
    except Exception: return torch.cuda.amp.GradScaler()

# ---------------- load BASIL (used for the overlap check and stage 2) ----------------
need = [f"{DATA}/{n}_corrected.csv" for n in ["train", "validation", "test"]]
if any(not os.path.exists(p) for p in need): raise FileNotFoundError(f"Missing BASIL files: {need}")
tr, va, te = [pd.read_csv(f"{DATA}/{n}_corrected.csv") for n in ["train", "validation", "test"]]
for d in (tr, va, te): d["target_sentence"] = d["target_sentence"].fillna("")
y_tr, y_va, y_te = tr.label.values, va.label.values, te.label.values
print(f"BASIL: train {len(tr)} ({y_tr.sum()} biased) | validation {len(va)} ({y_va.sum()}) | test {len(te)} ({y_te.sum()})")

# ============================================================
# BABE: select sentences whose bias is carried by loaded words (same rule as before)
# ============================================================
if not os.path.exists(BABE_PATH): raise FileNotFoundError(f"Upload final_labels_SG2.xlsx to {BABE_PATH}")
df = pd.read_excel(BABE_PATH)
df = df[df["label_bias"].isin(["Biased", "Non-biased"])].copy()
def parse(c):
    if isinstance(c, str) and c.strip().startswith("["):
        try: return [str(w).strip().lower() for w in ast.literal_eval(c)]
        except Exception: return []
    return []
df["marked"] = df["biased_words"].apply(parse)
df["toks"]   = df["text"].apply(lambda s: re.findall(r"[a-z][a-z'\-]+", str(s).lower()))
STOP = set("""a an the and or but of to in on at by for with from up out down off over as is are was were be been being
it its this that these those he she they them his her their we you i not no nor so than then there here who whom which what
when where how all any some more most other such only own same too very can will just do does did has have had said says say
would could should may might about into after before between through during under again further once""".split())
appear, marked = Counter(), Counter()
for toks, mk, lab in zip(df.toks, df.marked, df.label_bias):
    for w in set(toks): appear[w] += 1
    if lab == "Biased":
        for w in set(mk): marked[w] += 1
def is_cue(w):
    if w in STOP or len(w) < 3: return False
    if appear[w] >= MIN_COUNT: return marked[w] / appear[w] >= RATE_MIN
    return True
df["cue_words"] = df.marked.apply(lambda ws: [w for w in ws if is_cue(w)])
df["n_cues"] = df.cue_words.str.len()
pos = df[(df.label_bias == "Biased") & (df.n_cues >= 1)].copy()
neg = df[df.label_bias == "Non-biased"].copy()
if MAX_PER_OUTLET: pos = pos.sort_values("n_cues", ascending=False).groupby("outlet", group_keys=False).head(MAX_PER_OUTLET)
babe = pd.concat([pos.assign(label=1), neg.assign(label=0)], ignore_index=True).rename(columns={"text": "target_sentence"})
babe = babe[["target_sentence", "label", "outlet"]].drop_duplicates("target_sentence").reset_index(drop=True)

# leakage check: BABE sentences identical to BASIL validation/test sentences are removed
norm = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()
held = set(map(norm, pd.concat([va.target_sentence, te.target_sentence])))
dup = babe.target_sentence.map(lambda s: norm(s) in held)
print(f"BABE sentences identical to BASIL validation/test sentences: {int(dup.sum())} (removed)")
babe = babe[~dup].reset_index(drop=True)
print(f"BABE selected: {len(babe)} sentences | lexical (1): {int(babe.label.sum())} | non-biased (0): {int((babe.label==0).sum())}")
print("outlet share among selected positives:", babe[babe.label == 1].outlet.value_counts(normalize=True).round(2).head(4).to_dict())

# ============================================================
# BABE fine-tuning (10% of BABE held out only to pick the best epoch)
# ============================================================
B_DIR = f"{OUT}/babe_checkpoint"
if os.path.exists(f"{B_DIR}/config.json") and not OVERWRITE:
    print("BABE-stage checkpoint found on Drive -> skipping:", B_DIR)
else:
    b_tr, b_va = train_test_split(babe, test_size=0.1, stratify=babe.label, random_state=0)
    enc = lambda d: tok(list(d.target_sentence), truncation=True, max_length=B_MAX_LEN)
    ds_btr, ds_bva = DS(enc(b_tr), b_tr.label.values), DS(enc(b_va), b_va.label.values)
    set_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(INIT, num_labels=2).to(device)
    steps = int(np.ceil(len(ds_btr) / B_BATCH)) * B_EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=B_LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); scaler = new_scaler()
    best_auc, best_state = -1, None
    for ep in range(B_EPOCHS):
        model.train(); total, n = 0.0, 0
        for b in DataLoader(ds_btr, batch_size=B_BATCH, shuffle=True, collate_fn=collate):
            b = {k: v.to(device) for k, v in b.items()}; labels = b.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**b).logits
            loss = F.cross_entropy(logits.float(), labels)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            total += loss.item(); n += 1
        auc = roc_auc_score(b_va.label.values, predict(model, ds_bva))
        print(f"   [BABE stage] epoch {ep+1}/{B_EPOCHS} | train loss {total/n:.3f} | held-out BABE ROC-AUC {auc:.3f}")
        if auc > best_auc: best_auc, best_state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.save_pretrained(B_DIR); print("Saved BABE-stage checkpoint ->", B_DIR)
    del model, opt; torch.cuda.empty_cache()

# ============================================================
# STAGE 2: corrected BASIL, starting from the BABE-stage checkpoint
# ============================================================
enc2 = lambda d: tok(list(d.target_sentence), truncation=True, max_length=MAX_LEN)
ds_tr, ds_va, ds_te = DS(enc2(tr), y_tr), DS(enc2(va), y_va), DS(enc2(te), y_te)

def train_stage2(seed):
    set_seed(seed); rng = np.random.default_rng(seed)
    model = AutoModelForSequenceClassification.from_pretrained(B_DIR, num_labels=2).to(device)
    pos_idx, neg_idx = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
    def epoch_idx():
        idx = np.concatenate([pos_idx, rng.choice(neg_idx, size=min(len(neg_idx), NEG_PER_POS * len(pos_idx)), replace=False)])
        rng.shuffle(idx); return idx
    steps_per_epoch = int(np.ceil(len(epoch_idx()) / BATCH))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps_per_epoch * EPOCHS), steps_per_epoch * EPOCHS); scaler = new_scaler()
    best_ap, best_state = -1, None
    for ep in range(EPOCHS):
        model.train(); total = 0.0
        for b in DataLoader(Subset(ds_tr, epoch_idx()), batch_size=BATCH, collate_fn=collate):
            b = {k: v.to(device) for k, v in b.items()}; labels = b.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**b).logits
            loss = F.cross_entropy(logits.float(), labels)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); total += loss.item()
        ap = average_precision_score(y_va, predict(model, ds_va))
        print(f"   [BASIL seed {seed}] epoch {ep+1}/{EPOCHS} | train loss {total/steps_per_epoch:.3f} | BASIL validation PR-AUC {ap:.3f}")
        if ap > best_ap: best_ap, best_state = ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pv, pt = predict(model, ds_va), predict(model, ds_te)
    del model, opt; torch.cuda.empty_cache()
    return pv, pt

mode = RUN_NAME.replace("xlm-roberta-base_", "")
rows, pvs, pts = [], [], []
for seed in SEEDS:
    f = f"{OUT}/{mode}_seed{seed}_probs.npz"
    if os.path.exists(f) and not OVERWRITE:
        z = np.load(f); pv, pt = z["val"], z["test"]; print(f"[seed {seed}] loaded from disk")
    else:
        pv, pt = train_stage2(seed); np.savez(f, val=pv, test=pt)
    t, _ = pick_threshold(y_va, pv); rows.append(dict(mode=mode, seed=seed, **metrics(y_te, pt, t))); pvs.append(pv); pts.append(pt)
pv, pt = np.mean(pvs, 0), np.mean(pts, 0)
t, _ = pick_threshold(y_va, pv); m = metrics(y_te, pt, t)
f1_ci, ap_ci = bootstrap_by_story(y_te, pt, t, te.story_id.values)
rows.append(dict(mode=mode, seed="ensemble", **m, f1_ci=f"{f1_ci[0]:.3f}-{f1_ci[1]:.3f}", pr_auc_ci=f"{ap_ci[0]:.3f}-{ap_ci[1]:.3f}"))
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/results.csv", index=False)
pd.DataFrame({"sentenceId": te.sentenceId, "label": y_te, "prob": pt, "pred": (pt >= t).astype(int)}).to_csv(f"{OUT}/ensemble_test_predictions.csv", index=False)

show = ["mode", "seed", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "fp", "fn"]
print(f"\n=== {mode} (test set; threshold chosen on validation) ==="); print(res[show].round(3).to_string(index=False))
print("\nensemble 95% intervals (resampling stories):"); print(res[res.seed == "ensemble"][["f1", "f1_ci", "pr_auc", "pr_auc_ci"]].round(3).to_string(index=False))

# ---- side-by-side with the earlier runs (mean over seeds, test set) ----
comp = {}
for label, folder in [("BASIL only", "xlm-roberta-base_sample5"), ("Anno -> BASIL", "xlm-roberta-base_anno_then_basil"), (mode.replace("_", " -> ").upper().replace("->", "->"), RUN_NAME)]:
    p = f"{BASE}/runs/{folder}/results.csv"
    if os.path.exists(p):
        r = pd.read_csv(p); r = r[r.seed.astype(str) != "ensemble"]
        if "mode" in r and folder == "xlm-roberta-base_sample5": r = r[r["mode"] == "target"]
        comp[label] = r[["precision", "recall", "f1", "pr_auc", "roc_auc"]].astype(float).mean()
print("\n=== comparison (mean over seeds, BASIL test set) ==="); print(pd.DataFrame(comp).round(3).to_string())
print("\nSaved to:", OUT)

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 7 | exec None
# ====================================================================================================
# ============================================================
# Paired story-level bootstrap between setups (CPU only, no training).
# Uses the saved test probabilities (average of the 3 seeds) from each run folder.
# Reports the DIFFERENCE in ROC-AUC and PR-AUC (setup B minus setup A) with a 95% interval.
# If the interval does not include 0, the difference is unlikely to be seed/sampling luck.
# ============================================================
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = "/content/gdrive/MyDrive/BASIL_datasets"
SEEDS = [42, 43, 44]
RUNS = {   # name: (folder, file prefix)
    "BASIL only":          ("xlm-roberta-base_sample5",           "target"),
    "BABE -> BASIL":       ("xlm-roberta-base_babe_basil",        "babe_basil"),
    "Anno -> BASIL":       ("xlm-roberta-base_anno_then_basil",   "anno_then_basil"),
    "Anno -> BABE -> BASIL": ("xlm-roberta-base_anno_babe_basil", "anno_babe_basil"),
}
PAIRS = [("BASIL only", "Anno -> BASIL"), ("BASIL only", "BABE -> BASIL"), ("BASIL only", "Anno -> BABE -> BASIL"),
         ("BABE -> BASIL", "Anno -> BASIL"), ("Anno -> BASIL", "Anno -> BABE -> BASIL")]
N_BOOT = 2000

te = pd.read_csv(f"{BASE}/corrected/test_corrected.csv")
y, story = te.label.values, te.story_id.values
P = {}
for name, (folder, pre) in RUNS.items():
    fs = [f"{BASE}/runs/{folder}/{pre}_seed{s}_probs.npz" for s in SEEDS]
    if all(os.path.exists(f) for f in fs): P[name] = np.mean([np.load(f)["test"] for f in fs], 0)
    else: print("missing, skipped:", name)

idx_of = {s: np.where(story == s)[0] for s in np.unique(story)}; keys = list(idx_of)
rng = np.random.default_rng(0)
boots = [np.concatenate([idx_of[k] for k in rng.choice(keys, size=len(keys), replace=True)]) for _ in range(N_BOOT)]
boots = [b for b in boots if y[b].sum() > 0]

rows = []
for name, p in P.items():
    rows.append(dict(setup=name, roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p)))
print("=== ensemble of 3 seeds, test set ==="); print(pd.DataFrame(rows).round(3).to_string(index=False))

out = []
for a, b in PAIRS:
    if a not in P or b not in P: continue
    d_roc = np.array([roc_auc_score(y[i], P[b][i]) - roc_auc_score(y[i], P[a][i]) for i in boots])
    d_ap  = np.array([average_precision_score(y[i], P[b][i]) - average_precision_score(y[i], P[a][i]) for i in boots])
    ci = lambda d: f"{d.mean():+.3f} [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]"
    out.append(dict(comparison=f"{b}  minus  {a}", roc_auc_diff=ci(d_roc), pr_auc_diff=ci(d_ap),
                    roc_clear=bool(np.percentile(d_roc,2.5) > 0 or np.percentile(d_roc,97.5) < 0),
                    pr_clear=bool(np.percentile(d_ap,2.5) > 0 or np.percentile(d_ap,97.5) < 0)))
print("\n=== paired story-level bootstrap: difference (mean [95% interval]) ===")
print(pd.DataFrame(out).to_string(index=False))
print("\nclear = the interval excludes 0.")

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 8 | exec None
# ====================================================================================================
# ============================================================
# WORD-LEVEL (span) TRAINING on BASIL's annotated phrases  +  sentence classification  (multi-task)
#   One shared encoder, two heads:
#     sentence head : biased / neutral                      (this is the score we report, as before)
#     word head     : is this word-piece inside an annotated biased phrase?  (uses BASIL `phrase_spans`)
#   START = "anno_babe" : start from the saved Anno-lexical -> BABE checkpoint   (main)
#   START = "base"      : start from plain xlm-roberta-base                       (ablation: spans alone)
# BASIL validation/test are never used for training. Epoch/threshold chosen on BASIL VALIDATION only.
# ============================================================
import os, json, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix, accuracy_score, f1_score)

# ---------------- settings ----------------
START       = "anno_babe"                # "anno_babe" (main) or "base" (ablation)
DRIVE_MOUNT = "/content/gdrive"
MODEL_NAME  = "xlm-roberta-base"
SEEDS       = [42, 43, 44]
EPOCHS, BATCH, LR, MAX_LEN = 8, 16, 1e-5, 256   # same as the other BASIL runs
NEG_PER_POS = 5
LAMBDA_TOK  = 1.0                        # weight of the word-level loss
TOK_POS_W   = 5.0                        # biased word-pieces are rare, so weight them more in the word loss
OVERWRITE   = False
RUN_NAME    = "xlm-roberta-base_" + ("span_anno_babe" if START == "anno_babe" else "span_base")

from google.colab import drive
drive.mount(DRIVE_MOUNT)
BASE = f"{DRIVE_MOUNT}/MyDrive/BASIL_datasets"; DATA = f"{BASE}/corrected"; OUT = f"{BASE}/runs/{RUN_NAME}"
INIT = f"{BASE}/runs/xlm-roberta-base_anno_babe_basil/babe_checkpoint" if START == "anno_babe" else MODEL_NAME
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| run:", RUN_NAME, "| start from:", INIT)
if device == "cpu": raise SystemExit("No GPU: connect to a GPU runtime first.")
if START == "anno_babe" and not os.path.exists(f"{INIT}/config.json"): raise FileNotFoundError(f"Run the BABE-stage cell first (missing {INIT}).")
os.makedirs(OUT, exist_ok=True)

# ---------------- helpers ----------------
def pick_threshold(y, p):
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.linspace(0.02, 0.98, 97), 2):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1 + 1e-12: best_t, best_f1 = float(t), f1
    return best_t, best_f1

def metrics(y, p, t):
    pred = (p >= t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=t, precision=pr[0], recall=rc[0], f1=f1[0],
                macro_f1=f1_score(y, pred, average="macro", zero_division=0), accuracy=accuracy_score(y, pred),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), predicted_biased=int(pred.sum()))

def bootstrap_by_story(y, p, t, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed); g = pd.Series(groups).values
    idx_of = {s: np.where(g == s)[0] for s in np.unique(g)}; keys = list(idx_of)
    f1s, aps = [], []
    for _ in range(n):
        pick = np.concatenate([idx_of[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        if y[pick].sum() == 0: continue
        f1s.append(f1_score(y[pick], (p[pick] >= t).astype(int), zero_division=0)); aps.append(average_precision_score(y[pick], p[pick]))
    return np.percentile(f1s, [2.5, 97.5]), np.percentile(aps, [2.5, 97.5])

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def new_scaler():
    try:    return torch.amp.GradScaler("cuda")
    except Exception: return torch.cuda.amp.GradScaler()

# ---------------- data ----------------
need = [f"{DATA}/{n}_corrected.csv" for n in ["train", "validation", "test"]]
if any(not os.path.exists(p) for p in need): raise FileNotFoundError(f"Missing BASIL files: {need}")
tr, va, te = [pd.read_csv(f"{DATA}/{n}_corrected.csv") for n in ["train", "validation", "test"]]
for d in (tr, va, te): d["target_sentence"] = d["target_sentence"].fillna("")
y_tr, y_va, y_te = tr.label.values, va.label.values, te.label.values
print(f"BASIL: train {len(tr)} ({y_tr.sum()} biased) | validation {len(va)} ({y_va.sum()}) | test {len(te)} ({y_te.sum()})")

def parse_spans(v):
    try: return [(int(s), int(e)) for s, e, *_ in json.loads(v)] if isinstance(v, str) and v.strip() else []
    except Exception: return []
for d in (tr, va, te): d["spans"] = d["phrase_spans"].apply(parse_spans)
print("biased sentences with at least one annotated phrase span:",
      f"train {int(((tr.label==1)&(tr.spans.str.len()>0)).sum())}/{int(tr.label.sum())}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
assert tok.is_fast, "A fast tokenizer is needed for character offsets."

def token_labels(offsets, spans):
    """1 if the word-piece overlaps an annotated phrase, 0 if not, -100 for special tokens (offset 0,0)."""
    out = []
    for (a, b) in offsets:
        if a == b == 0: out.append(-100)
        else: out.append(int(any(a < e and b > s for s, e in spans)))
    return out

def encode(df):
    enc = tok(list(df.target_sentence), truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    return [dict(input_ids=enc["input_ids"][i], attention_mask=enc["attention_mask"][i],
                 tok_labels=token_labels(enc["offset_mapping"][i], df.spans.iloc[i]), label=int(df.label.iloc[i]))
            for i in range(len(df))]

class DS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    L = max(len(x["input_ids"]) for x in batch)
    ids  = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    tl   = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"]); ids[i, :n] = torch.tensor(x["input_ids"]); mask[i, :n] = 1; tl[i, :n] = torch.tensor(x["tok_labels"])
    return dict(input_ids=ids, attention_mask=mask, tok_labels=tl, labels=torch.tensor([x["label"] for x in batch]))

class SpanModel(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.enc = AutoModel.from_pretrained(init)            # loads the encoder from either a base or a saved classifier checkpoint
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1); self.sent = nn.Linear(h, 2); self.word = nn.Linear(h, 2)
    def forward(self, input_ids, attention_mask):
        H = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        H = self.drop(H)
        return self.sent(H[:, 0]), self.word(H)

@torch.no_grad()
def predict(model, ds):
    """returns sentence prob, max word prob, and pointing hits (is the top word inside a gold phrase?)"""
    model.eval(); ps, pw, hit, has = [], [], [], []
    for b in DataLoader(ds, batch_size=64, collate_fn=collate):
        ids, m = b["input_ids"].to(device), b["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            ls, lw = model(ids, m)
        ps.append(torch.softmax(ls.float(), -1)[:, 1].cpu().numpy())
        w = torch.softmax(lw.float(), -1)[..., 1].cpu(); tl = b["tok_labels"]
        valid = tl != -100
        w_valid = torch.where(valid, w, torch.full_like(w, -1.0))
        pw.append(w_valid.max(1).values.clamp(min=0).numpy())
        top = w_valid.argmax(1)
        hit.append((tl[torch.arange(len(top)), top] == 1).numpy()); has.append(((tl == 1).any(1)).numpy())
    return np.concatenate(ps), np.concatenate(pw), np.concatenate(hit), np.concatenate(has)

def train_one(seed, ds_tr, ds_va):
    set_seed(seed); rng = np.random.default_rng(seed)
    model = SpanModel(INIT).to(device)
    pos_idx, neg_idx = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
    def epoch_idx():
        idx = np.concatenate([pos_idx, rng.choice(neg_idx, size=min(len(neg_idx), NEG_PER_POS * len(pos_idx)), replace=False)])
        rng.shuffle(idx); return idx
    steps_per_epoch = int(np.ceil(len(epoch_idx()) / BATCH))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps_per_epoch * EPOCHS), steps_per_epoch * EPOCHS); scaler = new_scaler()
    wtok = torch.tensor([1.0, TOK_POS_W], device=device)
    best_ap, best_state = -1, None
    for ep in range(EPOCHS):
        model.train(); total = 0.0
        for b in DataLoader(Subset(ds_tr, epoch_idx()), batch_size=BATCH, collate_fn=collate):
            ids, m, tl, labels = b["input_ids"].to(device), b["attention_mask"].to(device), b["tok_labels"].to(device), b["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                ls, lw = model(ids, m)
            loss = F.cross_entropy(ls.float(), labels) + LAMBDA_TOK * F.cross_entropy(lw.float().reshape(-1, 2), tl.reshape(-1), weight=wtok, ignore_index=-100)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); total += loss.item()
        ap = average_precision_score(y_va, predict(model, ds_va)[0])
        print(f"   [{RUN_NAME} seed {seed}] epoch {ep+1}/{EPOCHS} | train loss {total/steps_per_epoch:.3f} | BASIL validation PR-AUC {ap:.3f}")
        if ap > best_ap: best_ap, best_state = ap, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model

ds_tr, ds_va, ds_te = DS(encode(tr)), DS(encode(va)), DS(encode(te))

# ---------------- run ----------------
mode = RUN_NAME.replace("xlm-roberta-base_", "")
rows, store = [], {"sent": [], "comb": []}; pvs = {"sent": [], "comb": []}; hits = []
for seed in SEEDS:
    f = f"{OUT}/{mode}_seed{seed}_probs.npz"
    if os.path.exists(f) and not OVERWRITE:
        z = np.load(f); d = {k: z[k] for k in z.files}; print(f"[seed {seed}] loaded from disk")
    else:
        model = train_one(seed, ds_tr, ds_va)
        sv, wv, _, _ = predict(model, ds_va); st, wt, hit, has = predict(model, ds_te)
        d = dict(val=sv, test=st, val_comb=0.5 * (sv + wv), test_comb=0.5 * (st + wt), hit=hit, has=has)
        np.savez(f, **d); del model; torch.cuda.empty_cache()
    for name, kv, kt in [("sent", "val", "test"), ("comb", "val_comb", "test_comb")]:
        t, _ = pick_threshold(y_va, d[kv]); rows.append(dict(mode=f"{mode}:{name}", seed=seed, **metrics(y_te, d[kt], t)))
        pvs[name].append(d[kv]); store[name].append(d[kt])
    hits.append((d["hit"][d["has"] & (y_te == 1)].mean()))
for name in ["sent", "comb"]:
    pv, pt = np.mean(pvs[name], 0), np.mean(store[name], 0)
    t, _ = pick_threshold(y_va, pv); m = metrics(y_te, pt, t)
    f1_ci, ap_ci = bootstrap_by_story(y_te, pt, t, te.story_id.values)
    rows.append(dict(mode=f"{mode}:{name}", seed="ensemble", **m, f1_ci=f"{f1_ci[0]:.3f}-{f1_ci[1]:.3f}", pr_auc_ci=f"{ap_ci[0]:.3f}-{ap_ci[1]:.3f}"))
    if name == "sent":
        pd.DataFrame({"sentenceId": te.sentenceId, "label": y_te, "prob": pt, "pred": (pt >= t).astype(int)}).to_csv(f"{OUT}/ensemble_test_predictions.csv", index=False)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/results.csv", index=False)

show = ["mode", "seed", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "fp", "fn"]
print("\n=== PRIMARY score = sentence head (':sent'). ':comb' = average of sentence prob and top word prob (extra, reported separately) ===")
print(res[show].round(3).to_string(index=False))
print("\nensemble 95% intervals (resampling stories):"); print(res[res.seed == "ensemble"][["mode", "f1", "f1_ci", "pr_auc", "pr_auc_ci"]].round(3).to_string(index=False))
print(f"\nPOINTING: on test sentences that are biased, the word the model scores highest lies inside an annotated phrase in "
      f"{np.mean(hits):.0%} of cases (mean over seeds).")

comp = {}
for label, folder, filt in [("BASIL only", "xlm-roberta-base_sample5", ("mode", "target")), ("Anno -> BASIL", "xlm-roberta-base_anno_then_basil", None),
                            ("BABE -> BASIL", "xlm-roberta-base_babe_basil", None), ("Anno -> BABE -> BASIL", "xlm-roberta-base_anno_babe_basil", None)]:
    p = f"{BASE}/runs/{folder}/results.csv"
    if os.path.exists(p):
        r = pd.read_csv(p); r = r[r.seed.astype(str) != "ensemble"]
        if filt: r = r[r[filt[0]] == filt[1]]
        comp[label] = r[["precision", "recall", "f1", "pr_auc", "roc_auc"]].astype(float).mean()
r = res[(res.seed != "ensemble") & (res["mode"] == f"{mode}:sent")]
comp[f"{mode} (sentence head)"] = r[["precision", "recall", "f1", "pr_auc", "roc_auc"]].astype(float).mean()
print("\n=== comparison (mean over seeds, BASIL test set) ==="); print(pd.DataFrame(comp).round(3).to_string())
print("\nSaved to:", OUT)

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 9 | exec None
# ====================================================================================================
# ============================================================
# ERROR ANALYSIS (CPU only, no training).
# Uses the saved test predictions of the best setup (Anno-lexical -> BABE -> BASIL, average of 3 seeds)
# and the corrected BASIL test file. Answers: which biased sentences are missed, which neutral ones are flagged?
# ============================================================
import os, re, json
import numpy as np, pandas as pd
from collections import Counter

DRIVE_MOUNT = "/content/gdrive"
RUN = "xlm-roberta-base_anno_babe_basil"          # change to analyse another setup
try:
    from google.colab import drive; drive.mount(DRIVE_MOUNT)
except Exception: pass
BASE = f"{DRIVE_MOUNT}/MyDrive/BASIL_datasets"
pred = pd.read_csv(f"{BASE}/runs/{RUN}/ensemble_test_predictions.csv")
te   = pd.read_csv(f"{BASE}/corrected/test_corrected.csv")
df = te.merge(pred[["sentenceId", "prob", "pred"]], on="sentenceId", how="left")
assert df.prob.notna().all(), "prediction file does not match the test file"
df["target_sentence"] = df.target_sentence.fillna("")
df["outcome"] = np.select([(df.label == 1) & (df.pred == 1), (df.label == 1) & (df.pred == 0), (df.label == 0) & (df.pred == 1)],
                          ["TP (caught)", "FN (missed)", "FP (false alarm)"], "TN")
print("Setup:", RUN); print(df.outcome.value_counts().to_string())
print(f"precision {((df.outcome=='TP (caught)').sum() / max(1,(df.pred==1).sum())):.3f} | recall {((df.outcome=='TP (caught)').sum() / (df.label==1).sum()):.3f}")

# ---------- simple sentence features ----------
df["words"]   = df.target_sentence.str.split().str.len()
df["has_quote"] = df.target_sentence.str.contains(r"[\"“”]")
df["first_sentence"] = df.sentence_position.astype(float) <= 1 if "sentence_position" in df else False
def phrases(x): return [p.strip() for p in str(x).split("|") if p.strip() and str(x) != "nan"]
df["phr"] = df.lexical_bias_phrases.apply(phrases)
df["n_phr"] = df.phr.str.len()

def rate_table(col, bins=None, labels=None):
    d = df.copy()
    if bins is not None: d[col] = pd.cut(d[col], bins=bins, labels=labels)
    g = d.groupby(col, observed=True)
    out = pd.DataFrame({"sentences": g.size(), "biased": g.label.sum(), "caught": g.apply(lambda x: ((x.label == 1) & (x.pred == 1)).sum()),
                        "false_alarms": g.apply(lambda x: ((x.label == 0) & (x.pred == 1)).sum())})
    out["recall"] = (out.caught / out.biased.replace(0, np.nan)).round(2)
    out["false_alarm_rate"] = (out.false_alarms / (out.sentences - out.biased)).round(3)
    return out

print("\n=== by quotation (sentence contains quotation marks) ==="); print(rate_table("has_quote").to_string())
print("\n=== by sentence length (words) ==="); print(rate_table("words", [0, 12, 20, 30, 1000], ["<=12", "13-20", "21-30", ">30"]).to_string())
print("\n=== by number of annotated biased phrases (biased sentences only counted in recall) ==="); print(rate_table("n_phr").to_string())
if "source" in df: print("\n=== by outlet ==="); print(rate_table("source").to_string())

# ---------- phrase level ----------
rows = []
for _, r in df[df.label == 1].iterrows():
    for p in r.phr: rows.append(dict(phrase=p.lower(), words=len(p.split()), caught=int(r.pred == 1), prob=r.prob))
ph = pd.DataFrame(rows)
if len(ph):
    ph["length"] = pd.cut(ph.words, [0, 1, 2, 4, 100], labels=["1 word", "2 words", "3-4 words", "5+ words"])
    print("\n=== recall by phrase length (a phrase counts as caught if its sentence was flagged) ===")
    print(ph.groupby("length", observed=True).caught.agg(["count", "mean"]).rename(columns={"count": "phrases", "mean": "recall"}).round(2).to_string())

# ---------- words that go with the errors ----------
STOP = set("the a an and or of to in on at by for with from as is are was were be been it its this that he she they his her their we you i not no but so than then there who which what when where how all any some more most other said says say would could should may might about into after before".split())
tok = lambda s: set(w for w in re.findall(r"[a-z][a-z'\-]+", s.lower()) if w not in STOP and len(w) > 2)
def overrep(group, rest, min_count=4, top=12):
    cg, cr = Counter(), Counter()
    for s in group.target_sentence: cg.update(tok(s))
    for s in rest.target_sentence: cr.update(tok(s))
    ng, nr = len(group), len(rest); out = []
    for w, c in cg.items():
        if c >= min_count: out.append((w, c, round(np.log(((c + .5) / (ng + 1)) / ((cr[w] + .5) / (nr + 1))), 2)))
    return sorted(out, key=lambda x: -x[2])[:top]
fp, tn = df[df.outcome == "FP (false alarm)"], df[df.outcome == "TN"]
fn, tp = df[df.outcome == "FN (missed)"], df[df.outcome == "TP (caught)"]
print("\n=== words over-represented in FALSE ALARMS vs correctly ignored neutral sentences (word, count, log-ratio) ===")
print(overrep(fp, tn))
print("\n=== words over-represented in MISSED biased sentences vs CAUGHT ones (word, count, log-ratio; few sentences, read with care) ===")
print(overrep(fn, tp, min_count=3))

# ---------- sentences to read ----------
pd.set_option("display.max_colwidth", 220); pd.set_option("display.width", 250)
print("\n=== 12 most confident FALSE ALARMS (read these: are they truly neutral?) ===")
print(fp.sort_values("prob", ascending=False).head(12)[["prob", "target_sentence"]].round(2).to_string(index=False))
print("\n=== 12 MISSED biased sentences, lowest probability first (with the annotated phrase) ===")
print(fn.sort_values("prob").head(12).assign(phrase=lambda d: d.phr.str.join(" | "))[["prob", "phrase", "target_sentence"]].round(2).to_string(index=False))
print("\n=== 8 CAUGHT biased sentences (for the report as positive examples) ===")
print(tp.sort_values("prob", ascending=False).head(8).assign(phrase=lambda d: d.phr.str.join(" | "))[["prob", "phrase", "target_sentence"]].round(2).to_string(index=False))

out = f"{BASE}/runs/{RUN}/error_analysis_sentences.csv"
df[df.outcome != "TN"].assign(phrases=lambda d: d.phr.str.join(" | ")).sort_values(["outcome", "prob"], ascending=[True, False])[
    ["outcome", "prob", "source", "target_sentence", "phrases"]].to_csv(out, index=False)
print("\nSaved every error with its text to:", out)

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 10 | exec 2
# ====================================================================================================
# ============================================================
# SPLIT AUDIT (CPU only, no training). Run in Colab after mounting Drive.
# 1) writes split_manifest.csv (article / story IDs only, no text)
# 2) reports how many stories (triplets) appear in more than one split
# 3) re-scores the saved test predictions on the test sentences whose story
#    has NO article in training ("story-disjoint"), to see if shared stories inflate results
# ============================================================
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = "/content/gdrive/MyDrive/BASIL_datasets"          # change if your Drive path differs
CORR = f"{BASE}/corrected"
SEEDS = [42, 43, 44]
RUNS = {"BASIL only": ("xlm-roberta-base_sample5", "target"),
        "BABE -> BASIL": ("xlm-roberta-base_babe_basil", "babe_basil"),
        "Anno -> BASIL": ("xlm-roberta-base_anno_then_basil", "anno_then_basil"),
        "Anno -> BABE -> BASIL": ("xlm-roberta-base_anno_babe_basil", "anno_babe_basil")}
N_BOOT = 1000

d = {n: pd.read_csv(f"{CORR}/{n}_corrected.csv") for n in ["train", "validation", "test"]}
for n, df in d.items(): df["split"] = n
al = pd.concat(d.values(), ignore_index=True)

# ---- 1. manifest ----
man = (al.groupby("articleId").agg(story_id=("story_id", "first"), source=("source", "first"),
        split=("split", "first"), n_split=("split", "nunique"), sentences=("label", "size"),
        biased=("label", "sum")).reset_index())
assert (man.n_split == 1).all(), "an article appears in two splits!"
if "fold" in al: man["fold"] = al.groupby("articleId").fold.first().values
man.drop(columns="n_split").to_csv(f"{CORR}/split_manifest.csv", index=False)
print("manifest written:", f"{CORR}/split_manifest.csv", "| articles:", len(man))

# ---- 2. counts ----
print("\narticles / stories / sentences / biased per split")
print(man.groupby("split").agg(articles=("articleId", "size"), stories=("story_id", "nunique"),
      sentences=("sentences", "sum"), biased=("biased", "sum")).to_string())
sp = man.groupby("story_id").split.agg(lambda s: tuple(sorted(s)))
n_in = man.groupby("story_id").split.nunique()
print("\nstories (triplets):", len(sp), "| entirely inside ONE split:", int((n_in == 1).sum()),
      "| spread over 2 or 3 splits:", int((n_in > 1).sum()))
print("\nlayout of each story (articles per split):")
print(man.groupby(["story_id", "split"]).size().unstack(fill_value=0).value_counts().to_string())
print("\noutlets per split:"); print(pd.crosstab(man.split, man.source).to_string())

# ---- 3. sensitivity on the test set ----
tr_st = set(man[man.split == "train"].story_id)
te = d["test"].copy()
te["shared"] = te.story_id.isin(tr_st)
print(f"\ntest sentences: {len(te)} (biased {int(te.label.sum())}) | from stories that also have a TRAIN article: "
      f"{int(te.shared.sum())} (biased {int(te[te.shared].label.sum())}) | story-disjoint: {int((~te.shared).sum())} "
      f"(biased {int(te[~te.shared].label.sum())})")
y, story = te.label.values, te.story_id.values
def boot_ci(p, mask):
    ids = np.where(mask)[0]; st = np.unique(story[ids]); idx = {s: ids[story[ids] == s] for s in st}
    rng = np.random.default_rng(0); r = []
    for _ in range(N_BOOT):
        b = np.concatenate([idx[k] for k in rng.choice(st, len(st))])
        if 0 < y[b].sum() < len(b): r.append(roc_auc_score(y[b], p[b]))
    return np.percentile(r, [2.5, 97.5])
rows = []
for name, (folder, pre) in RUNS.items():
    fs = [f"{BASE}/runs/{folder}/{pre}_seed{s}_probs.npz" for s in SEEDS]
    if not all(os.path.exists(f) for f in fs): print("missing:", name); continue
    p = np.mean([np.load(f)["test"] for f in fs], 0)
    for lab, m in [("all test", np.ones(len(te), bool)), ("shared-story", te.shared.values), ("story-disjoint", ~te.shared.values)]:
        if y[m].sum() == 0: continue
        lo, hi = boot_ci(p, m)
        rows.append(dict(setup=name, subset=lab, sentences=int(m.sum()), biased=int(y[m].sum()),
                         roc_auc=round(roc_auc_score(y[m], p[m]), 3), roc_95ci=f"[{lo:.3f}, {hi:.3f}]",
                         pr_auc=round(average_precision_score(y[m], p[m]), 3)))
print("\n=== ensemble of 3 seeds, by subset of the test set ==="); print(pd.DataFrame(rows).to_string(index=False))

# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 11 | exec 1
# ====================================================================================================

from pathlib import Path
import os, zipfile, hashlib, csv
from collections import Counter

# Your actual mounted Google Drive path
DRIVE = Path("/content/gdrive/MyDrive")

# Output bundle
ZIP_PATH = DRIVE / "LLM_Research_Audit_Bundle.zip"
MANIFEST_PATH = DRIVE / "LLM_Research_Audit_Manifest.csv"

# Relevant folders and file patterns
RELEVANT_FOLDER_NAMES = [
    "colab notebooks",
    "basil_datasets",
    "basil_datasets (1)"
]

RELEVANT_ROOT_FILES = {
    "lexical_bias_dataset_corrected.csv",
    "train_corrected.csv",
    "validation_corrected.csv",
    "test_corrected.csv"
}

EXTENSIONS = {
    ".ipynb", ".csv", ".tsv", ".json", ".jsonl",
    ".txt", ".log", ".yaml", ".yml", ".md",
    ".png", ".jpg", ".jpeg", ".svg", ".xlsx"
}

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints",
    "node_modules", "wandb", "cache", "caches"
}

MAX_FILE_MB = 75
MAX_TOTAL_MB = 400

if not DRIVE.exists():
    raise FileNotFoundError(
        f"Drive folder not found: {DRIVE}. Check your mount path."
    )

# Find relevant folders, including nested folders with these names
relevant_roots = []

for folder in DRIVE.rglob("*"):
    if folder.is_dir() and folder.name.lower() in RELEVANT_FOLDER_NAMES:
        relevant_roots.append(folder)

# Include relevant CSVs located directly in MyDrive
selected = set()

for name in RELEVANT_ROOT_FILES:
    p = DRIVE / name
    if p.is_file():
        selected.add(p)

# Include files inside relevant folders
for folder in relevant_roots:
    for current, dirs, files in os.walk(folder):
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDE_DIRS
            and not d.startswith("checkpoint-")
        ]

        for filename in files:
            p = Path(current) / filename
            if p.suffix.lower() in EXTENSIONS:
                try:
                    if p.stat().st_size <= MAX_FILE_MB * 1024**2:
                        selected.add(p)
                except OSError:
                    pass

# Also locate likely experiment outputs elsewhere in MyDrive.
# Only include files with experiment-related names, not all Drive files.
keywords = [
    "prediction", "predictions", "metric", "result",
    "evaluation", "confusion", "bootstrap", "split",
    "seed", "training", "validation", "test",
    "basil", "babe", "anno", "lexical", "bias"
]

for current, dirs, files in os.walk(DRIVE):
    dirs[:] = [
        d for d in dirs
        if d not in EXCLUDE_DIRS
        and d not in {"LLM_Research_Audit"}
        and not d.startswith("checkpoint-")
    ]

    for filename in files:
        p = Path(current) / filename

        if p.suffix.lower() not in EXTENSIONS:
            continue

        if not any(k in filename.lower() for k in keywords):
            continue

        try:
            if p.stat().st_size <= MAX_FILE_MB * 1024**2:
                selected.add(p)
        except OSError:
            pass

# Remove previous audit bundle and manifest if found
selected = {
    p for p in selected
    if p.resolve() not in {
        ZIP_PATH.resolve(), MANIFEST_PATH.resolve()
    }
}

# Generate manifest with hashes
manifest = []
for p in sorted(selected):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

        manifest.append({
            "drive_relative_path": str(p.relative_to(DRIVE)),
            "extension": p.suffix.lower(),
            "size_bytes": p.stat().st_size,
            "sha256": h.hexdigest()
        })
    except Exception as e:
        print("Could not inspect:", p, e)

with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "drive_relative_path", "extension",
            "size_bytes", "sha256"
        ]
    )
    writer.writeheader()
    writer.writerows(manifest)

# Create ZIP with a total size limit
total = 0
included = []
skipped = []

with zipfile.ZipFile(
    ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED
) as z:
    for p in sorted(selected):
        size = p.stat().st_size

        if total + size > MAX_TOTAL_MB * 1024**2:
            skipped.append((str(p), "Total bundle size limit"))
            continue

        try:
            arcname = str(p.relative_to(DRIVE))
            z.write(p, arcname)
            included.append(arcname)
            total += size
        except Exception as e:
            skipped.append((str(p), str(e)))

    z.write(MANIFEST_PATH, "AUDIT_FILE_MANIFEST.csv")

print("\n========== AUDIT BUNDLE ==========")
print("Drive:", DRIVE)
print("Notebooks/dataset folders found:")
for p in relevant_roots:
    print(" -", p.relative_to(DRIVE))

print("\nFiles included:", len(included))
print("Files skipped:", len(skipped))
print("ZIP size: %.2f MB" % (ZIP_PATH.stat().st_size / 1024**2))

print("\nIncluded file types:")
print(dict(Counter(Path(x).suffix.lower() for x in included)))

print("\nIncluded files:")
for x in included:
    print("✓", x)

if skipped:
    print("\nSkipped files:")
    for x, reason in skipped:
        print("-", x, "|", reason)

print("\nZIP LOCATION:")
print(ZIP_PATH)
print("\nMANIFEST LOCATION:")
print(MANIFEST_PATH)
print("==================================")


# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 12 | exec 2
# ====================================================================================================

from pathlib import Path
import os, zipfile, hashlib, csv

DRIVE = Path("/content/drive/MyDrive")
ZIP_PATH = DRIVE / "BASIL_Research_Audit_Only.zip"

NOTEBOOK_NAME = "BASIL_Lexical_Bias_Transfer_Experiments.ipynb"

# Only these file types are needed for the audit
EXTENSIONS = {
    ".ipynb", ".csv", ".tsv", ".json", ".jsonl",
    ".txt", ".log", ".yaml", ".yml", ".md",
    ".png", ".jpg", ".svg", ".xlsx"
}

# Relevant experiment names
KEYWORDS = [
    "basil", "babe", "anno", "lexical", "bias",
    "prediction", "predictions", "metric", "result",
    "evaluation", "confusion", "bootstrap", "split",
    "seed", "train_corrected", "validation_corrected",
    "test_corrected"
]

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints",
    "node_modules", "wandb", "cache", "caches"
}

selected = set()

# 1. Add the exact research notebook
for p in DRIVE.rglob(NOTEBOOK_NAME):
    if p.is_file():
        selected.add(p)

# 2. Add relevant dataset folders and files
for current, dirs, files in os.walk(DRIVE):
    dirs[:] = [
        d for d in dirs
        if d not in EXCLUDE_DIRS
        and not d.startswith("checkpoint-")
    ]

    current_path = Path(current)
    path_text = str(current_path.relative_to(DRIVE)).lower()

    for filename in files:
        p = current_path / filename
        name = filename.lower()

        if p.suffix.lower() not in EXTENSIONS:
            continue

        # Dataset folders or relevant experiment filenames
        in_dataset_folder = (
            "basil_datasets" in path_text
            or "basil_datasets" in name
        )

        relevant_name = any(k in name for k in KEYWORDS)

        if in_dataset_folder or relevant_name:
            try:
                if p.stat().st_size <= 100 * 1024**2:
                    selected.add(p)
            except OSError:
                pass

# 3. Ensure the four corrected CSVs are included if present
required_names = [
    "lexical_bias_dataset_corrected.csv",
    "train_corrected.csv",
    "validation_corrected.csv",
    "test_corrected.csv"
]

for required in required_names:
    matches = list(DRIVE.rglob(required))
    for p in matches:
        if p.is_file():
            selected.add(p)

# 4. Create ZIP and manifest
manifest = []
included = []

with zipfile.ZipFile(
    ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED
) as z:

    for p in sorted(selected):
        try:
            relative = str(p.relative_to(DRIVE))
            size = p.stat().st_size

            digest = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)

            z.write(p, relative)
            included.append(relative)

            manifest.append({
                "file": relative,
                "size_bytes": size,
                "sha256": digest.hexdigest()
            })

        except Exception as e:
            print("Skipped:", p, e)

    # Include manifest
    import io
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=["file", "size_bytes", "sha256"]
    )
    writer.writeheader()
    writer.writerows(manifest)

    z.writestr("AUDIT_MANIFEST.csv", output.getvalue())

print("\n========== RESEARCH-ONLY BUNDLE ==========")
print("ZIP:", ZIP_PATH)
print("Files included:", len(included))

for f in included:
    print("✓", f)

print("\nRequired-file check:")
for name in [NOTEBOOK_NAME] + required_names:
    found = any(Path(f).name == name for f in included)
    print(("✓ " if found else "MISSING: ") + name)

print("\nUpload this ZIP only after checking the file list.")
print("==========================================")


# ====================================================================================================
# SOURCE: <Drive>/Colab Notebooks/BASIL_Lexical_Bias_Transfer_Experiments.ipynb | cell 13 | exec None
# ====================================================================================================
# ==========================================================================================
# RESEARCH AUDIT EXPORT  (Google Colab, CPU is enough)
#
# What it does : collects the results that were ALREADY SAVED by your experiments into one small
#                folder (research_audit_export/) + a ZIP, so the report can be checked independently.
# What it never does: retrain a model, change an existing file, copy model weights, or invent a
#                value. Anything that was not saved is reported as MISSING.
#
# How to use   : paste this whole file into ONE Colab cell and run it. (Drive is mounted for you.)
#                Optional: upload any saved Colab notebooks (.ipynb), logs (.txt/.log) or scripts (.py)
#                from other accounts/sessions into /content/extra_inputs/ BEFORE running; they are used
#                for the code, the training logs and the settings.
# ==========================================================================================
import os, sys, re, io, json, glob, ast, shutil, platform, subprocess, datetime, zipfile, warnings
import numpy as np, pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix, accuracy_score, f1_score)
warnings.filterwarnings("ignore")

# ----------------------------- settings you may change -----------------------------------
MOUNT_CANDIDATES = ["/content/drive", "/content/gdrive"]      # tried in this order
BASE_OVERRIDE    = os.environ.get("AUDIT_BASE")               # leave None in Colab
EXTRA_DIR        = "/content/extra_inputs"                    # optional extra notebooks / logs / scripts
EXTRA_PATHS      = []                                         # or list full paths of extra files here
OUT_DIR          = os.environ.get("AUDIT_OUT", "/content/research_audit_export")
COPY_ZIP_TO_DRIVE = True                                      # also copy the ZIP to <BASE>/audit/
SEEDS_EXPECTED   = [42, 43, 44]
# -----------------------------------------------------------------------------------------

INV = []      # inventory rows
def inv(item, name, status, note=""):
    INV.append(dict(item=item, file=name, status=status, note=note))

# ----------------------------- locate Drive / experiment folder --------------------------
def locate_base():
    if BASE_OVERRIDE: return BASE_OVERRIDE
    for m in MOUNT_CANDIDATES:
        if os.path.isdir(f"{m}/MyDrive/BASIL_datasets"): return f"{m}/MyDrive/BASIL_datasets"
    try:
        from google.colab import drive
        for m in MOUNT_CANDIDATES:
            try:
                drive.mount(m)
                if os.path.isdir(f"{m}/MyDrive/BASIL_datasets"): return f"{m}/MyDrive/BASIL_datasets"
            except Exception as e:
                print(f"mount at {m} failed: {str(e)[:120]}")
    except Exception:
        pass
    raise SystemExit("Could not find BASIL_datasets on Drive. Mount the Drive account that holds the experiments "
                     "(the same one used for training) and re-run.")
BASE = locate_base(); MYDRIVE = os.path.dirname(BASE)
CORR = f"{BASE}/corrected"; RUNS = f"{BASE}/runs"
print("experiment folder:", BASE)
if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

# ----------------------------- the ORIGINAL helper functions (copied verbatim from the experiment code) ----
def pick_threshold(y, p):
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.linspace(0.02, 0.98, 97), 2):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1 + 1e-12: best_t, best_f1 = float(t), f1
    return best_t, best_f1

def metrics(y, p, t):
    pred = (p >= t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=t, precision=pr[0], recall=rc[0], f1=f1[0],
                macro_f1=f1_score(y, pred, average="macro", zero_division=0), accuracy=accuracy_score(y, pred),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), predicted_biased=int(pred.sum()))

# ----------------------------- A) split files (actual assignments, read only) ------------
def load_split(name):
    f = f"{CORR}/{name}_corrected.csv"
    if not os.path.exists(f): return None
    d = pd.read_csv(f); d["split"] = name; d["row_index"] = np.arange(len(d)); return d
S = {n: load_split(n) for n in ["train", "validation", "test"]}
if any(v is None for v in S.values()):
    raise SystemExit(f"Missing corrected split file(s) in {CORR}: {[n for n,v in S.items() if v is None]}")
va, te = S["validation"], S["test"]
y_va, y_te = va.label.values.astype(int), te.label.values.astype(int)
need_cols = ["sentenceId", "articleId", "story_id", "label"]
for n, d in S.items():
    miss = [c for c in need_cols if c not in d.columns]
    if miss: print(f"WARNING: {n}_corrected.csv lacks columns {miss}")

# H) split_manifest.csv
man = pd.concat(S.values(), ignore_index=True)
keep = [c for c in ["sentenceId", "articleId", "story_id", "split", "label", "row_index", "source", "fold"] if c in man.columns]
mf = man[keep].rename(columns={"sentenceId": "sentence_id", "articleId": "article_id", "row_index": "row_index_in_split_file"})
mf.to_csv(f"{OUT_DIR}/split_manifest.csv", index=False)
inv("H", "split_manifest.csv", "EXPORTED", f"{len(mf)} sentences; read straight from the *_corrected.csv files; "
    "sentence_id repeats across articles, so use (article_id, sentence_id) or row_index_in_split_file as the unique key")
dup = int(mf.duplicated(["article_id", "sentence_id"]).sum()) if {"article_id", "sentence_id"} <= set(mf.columns) else -1
art_split = mf.groupby("article_id").split.nunique()
print(f"split manifest: {len(mf)} rows | duplicate (article,sentence) keys: {dup} | articles in >1 split: {int((art_split>1).sum())}")

# ----------------------------- locate saved prediction files ------------------------------
KNOWN = {   # (run folder, file prefix) -> setup name used in the report
    ("xlm-roberta-base_sample5", "target"): "BASIL only",
    ("xlm-roberta-base_babe_basil", "babe_basil"): "BABE -> BASIL",
    ("xlm-roberta-base_anno_then_basil", "anno_then_basil"): "Anno-lexical -> BASIL",
    ("xlm-roberta-base_anno_babe_basil", "anno_babe_basil"): "Anno-lexical -> BABE -> BASIL",
    ("xlm-roberta-base_span_anno_babe", "span_anno_babe"): "Word-level (sentence head)",
}
runs = {}   # (folder, prefix) -> {seed: dict(arrays)}
shape_problems = []
for folder in sorted(glob.glob(f"{RUNS}/*")):
    if not os.path.isdir(folder): continue
    fname = os.path.basename(folder)
    for f in sorted(glob.glob(f"{folder}/*_seed*_probs.npz")):
        m = re.match(r"^(?P<prefix>.+)_seed(?P<seed>\d+)_probs\.npz$", os.path.basename(f))
        if not m: continue
        z = np.load(f, allow_pickle=False); d = {k: z[k] for k in z.files}
        ok = True
        for key, n_expect in [("val", len(va)), ("test", len(te))]:
            if key not in d or len(d[key]) != n_expect:
                shape_problems.append(f"{f}: '{key}' has {None if key not in d else len(d[key])} rows, expected {n_expect}")
                ok = False
        if ok: runs.setdefault((fname, m["prefix"]), {})[int(m["seed"])] = d
print(f"prediction files found for {len(runs)} (run, mode) combinations")
for p in shape_problems: print("SHAPE PROBLEM (not exported, not guessed):", p)

# series inside one npz: plain classifiers -> val/test ; word-level run also has val_comb/test_comb
def series_of(d):
    s = [("", "val", "test")]
    if "val_comb" in d and "test_comb" in d: s.append((":comb", "val_comb", "test_comb"))
    return s

def read_saved_results(folder):
    f = f"{RUNS}/{folder}/results.csv"
    return pd.read_csv(f) if os.path.exists(f) else None

# ----------------------------- B, C, E, F : predictions and metrics -----------------------
test_rows, val_rows, met_rows, cm_rows, ens_check = [], [], [], [], []
series_store = {}   # (setup, series_name) -> dict(seeds=..., ens_val, ens_test, per-seed)
def add_pred(rows, df, label, setup, seed, prob, thr, split, src):
    out = pd.DataFrame({"model_name": label, "report_setup": setup, "seed": seed, "split": split,
                        "row_index": df.row_index.values, "sentence_id": df.sentenceId.values,
                        "article_id": df.articleId.values, "story_id": df.story_id.values if "story_id" in df else np.nan,
                        "true_label": df.label.values.astype(int), "predicted_probability": prob.astype(np.float64),
                        "predicted_label": (prob >= thr).astype(int), "decision_threshold": thr, "probability_source": src})
    rows.append(out)

for (folder, prefix), by_seed in sorted(runs.items()):
    setup = KNOWN.get((folder, prefix), "NOT IN REPORT (exploratory / earlier run)")
    saved = read_saved_results(folder)
    for suf, kv, kt in series_of(next(iter(by_seed.values()))):
        label = f"{folder}/{prefix}{suf}"
        pvs, pts, seeds_here = [], [], []
        for seed in sorted(by_seed):
            d = by_seed[seed]; pv, pt = d[kv], d[kt]
            thr, _ = pick_threshold(y_va, pv)
            add_pred(test_rows, te, label, setup, seed, pt, thr, "test", f"npz:{kt}")
            add_pred(val_rows, va, label, setup, seed, pv, thr, "validation", f"npz:{kv}")
            for split, y, p in [("test", y_te, pt), ("validation", y_va, pv)]:
                m = metrics(y, p, thr)
                met_rows.append(dict(model_name=label, report_setup=setup, seed=seed, split=split, result_source="recomputed_from_saved_probabilities", n=len(y), n_positive=int(y.sum()), **m))
            pvs.append(pv); pts.append(pt); seeds_here.append(seed)
        ens_v = ens_t = None
        if len(pvs) >= 2:
            ens_v, ens_t = np.mean(pvs, 0), np.mean(pts, 0)
            thr, _ = pick_threshold(y_va, ens_v)
            add_pred(test_rows, te, label, setup, "ensemble", ens_t, thr, "test", f"mean of seeds {seeds_here} (npz:{kt})")
            add_pred(val_rows, va, label, setup, "ensemble", ens_v, thr, "validation", f"mean of seeds {seeds_here} (npz:{kv})")
            for split, y, p in [("test", y_te, ens_t), ("validation", y_va, ens_v)]:
                m = metrics(y, p, thr)
                met_rows.append(dict(model_name=label, report_setup=setup, seed="ensemble", split=split, result_source=f"recomputed_from_saved_probabilities (mean of seeds {seeds_here})", n=len(y), n_positive=int(y.sum()), **m))
            # compare with the ensemble file the experiment itself wrote
            ef = f"{RUNS}/{folder}/{prefix}_ensemble_test_predictions.csv"
            if not os.path.exists(ef): ef = f"{RUNS}/{folder}/ensemble_test_predictions.csv"
            if suf == "" and os.path.exists(ef):
                e = pd.read_csv(ef)
                if len(e) == len(te):
                    same_ids = bool((e.sentenceId.values == te.sentenceId.values).all()) if "sentenceId" in e else False
                    ens_check.append(dict(model_name=label, saved_file=ef.replace(BASE, "<BASE>"), same_sentence_order=same_ids,
                                          max_abs_prob_diff=float(np.abs(e.prob.values - ens_t).max()), labels_equal=bool((e.label.values == y_te).all())))
        # saved results.csv rows (the experiment's own numbers), for comparison
        if saved is not None:
            modes = {f"{prefix}", f"{prefix}{suf.replace(':comb', ':comb') }"}
            if suf == "" and prefix.startswith("span"): modes.add(f"{prefix}:sent")
            for _, r in saved[saved["mode"].astype(str).isin(modes | {f"{prefix}:sent" if suf == "" else f"{prefix}:comb"})].iterrows():
                if suf == "" and prefix.startswith("span") and str(r["mode"]).endswith(":comb"): continue
                if suf == ":comb" and not str(r["mode"]).endswith(":comb"): continue
                met_rows.append(dict(model_name=label, report_setup=setup, seed=r["seed"], split="test", result_source="saved_results_csv",
                                     n=len(te), n_positive=int(y_te.sum()),
                                     **{k: r[k] for k in ["threshold", "precision", "recall", "f1", "macro_f1", "accuracy", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp", "predicted_biased"] if k in r.index}))
        series_store[(setup, label)] = dict(label=label, seeds=seeds_here, ens_val=ens_v, ens_test=ens_t, pv=pvs, pt=pts)

tp = pd.concat(test_rows, ignore_index=True) if test_rows else None
vp = pd.concat(val_rows, ignore_index=True) if val_rows else None
if tp is not None:
    tp.to_csv(f"{OUT_DIR}/test_predictions.csv", index=False)
    inv("B", "test_predictions.csv", "EXPORTED", f"{len(tp)} rows; all seeds + ensembles found in runs/*/ *_probs.npz; mapped to sentence IDs by row order "
        f"(array length checked == {len(te)} test rows); probabilities are the saved values, not rounded")
    vp.to_csv(f"{OUT_DIR}/validation_predictions.csv", index=False)
    inv("C", "validation_predictions.csv", "EXPORTED", f"{len(vp)} rows; the same arrays used by pick_threshold; threshold column = threshold chosen from them")
else:
    inv("B", "test_predictions.csv", "MISSING", "no *_seed*_probs.npz files found under runs/")
    inv("C", "validation_predictions.csv", "MISSING", "no *_seed*_probs.npz files found under runs/")

# compare recomputed numbers with the experiment's own results.csv
met = pd.DataFrame(met_rows)
if len(met):
    met = met.rename(columns={"f1": "F1", "macro_f1": "macro_F1", "roc_auc": "ROC_AUC", "pr_auc": "PR_AUC", "tp": "TP", "fp": "FP", "tn": "TN", "fn": "FN"})
    rec = met[(met.result_source.str.startswith("recomputed")) & (met.split == "test")]
    sav = met[met.result_source == "saved_results_csv"]
    match = []
    for _, r in rec.iterrows():
        s = sav[(sav.model_name == r.model_name) & (sav.seed.astype(str) == str(r.seed))]
        if s.empty: match.append("NO_SAVED_ROW"); continue
        s = s.iloc[0]
        ok = abs(float(s.threshold) - r.threshold) < 1e-9 and abs(float(s.ROC_AUC) - r.ROC_AUC) < 1e-6 and abs(float(s.PR_AUC) - r.PR_AUC) < 1e-6 and abs(float(s.F1) - r.F1) < 1e-6
        match.append("MATCH" if ok else "DIFFERENT")
    met["matches_saved_results_csv"] = ""
    met.loc[rec.index, "matches_saved_results_csv"] = match
    cols = ["model_name", "report_setup", "seed", "split", "result_source", "n", "n_positive", "threshold", "accuracy", "precision", "recall", "F1", "macro_F1", "ROC_AUC", "PR_AUC", "TP", "FP", "TN", "FN", "predicted_biased", "matches_saved_results_csv"]
    met[[c for c in cols if c in met.columns]].to_csv(f"{OUT_DIR}/metrics_per_run.csv", index=False)
    inv("E", "metrics_per_run.csv", "EXPORTED", "rows labelled recomputed_from_saved_probabilities (test and validation; validation rows are optimistic because the threshold was chosen on them) and saved_results_csv (the experiment's own results.csv); last column compares them")
    cm = met[met.result_source.str.startswith("recomputed")][["model_name", "report_setup", "seed", "split", "threshold", "TP", "FP", "TN", "FN"]]
    cm.to_csv(f"{OUT_DIR}/confusion_matrices.csv", index=False)
    inv("F", "confusion_matrices.csv", "EXPORTED", f"{len(cm)} rows (test + validation, per seed and ensemble); TP/FP/TN/FN from saved probabilities at the validation-selected threshold")
    for x, nm in [("E", "metrics_per_run.csv"), ("F", "confusion_matrices.csv")]: pass
else:
    inv("E", "metrics_per_run.csv", "MISSING", "no predictions to compute from"); inv("F", "confusion_matrices.csv", "MISSING", "no predictions")
if ens_check:
    pd.DataFrame(ens_check).to_csv(f"{OUT_DIR}/ensemble_file_check.csv", index=False)

# ----------------------------- G) bootstrap ----------------------------------------------
boot_rows = []
story_te = te.story_id.values if "story_id" in te else None
def story_index(story):
    idx_of = {s: np.where(story == s)[0] for s in np.unique(story)}; return idx_of, list(idx_of)
def draw(story, n, seed=0):
    idx_of, keys = story_index(story); rng = np.random.default_rng(seed)
    return [np.concatenate([idx_of[k] for k in rng.choice(keys, size=len(keys), replace=True)]) for _ in range(n)]

# (1) the interval columns the experiment itself saved (f1_ci / pr_auc_ci in results.csv)
for (folder, prefix) in runs:
    sr = read_saved_results(folder)
    if sr is not None and "f1_ci" in sr:
        for _, r in sr[sr.seed.astype(str) == "ensemble"].iterrows():
            if not str(r["mode"]).startswith(prefix): continue
            for metric, col in [("F1", "f1_ci"), ("PR_AUC", "pr_auc_ci")]:
                if pd.notna(r.get(col)):
                    lo, hi = str(r[col]).split("-")
                    boot_rows.append(dict(model_pair=f"{folder}/{r['mode']} (ensemble, single model)", metric=metric, observed_difference="", confidence_interval_lower=float(lo), confidence_interval_upper=float(hi),
                                          number_of_bootstrap_iterations="1000 (default of bootstrap_by_story in the code)", random_seed="0 (default in the code)", resampling_unit="story_id", valid_resamples="NOT SAVED", discarded_resamples="NOT SAVED", source="saved_results_csv"))
# (2) recomputed from the saved ensemble probabilities (same procedure as paired_tests.py / bootstrap_by_story)
ENS = {}
for (setup, label), s in series_store.items():
    if setup in KNOWN.values() and s["ens_test"] is not None and not label.endswith(":comb") and sorted(s["seeds"]) == SEEDS_EXPECTED:
        ENS[setup] = (s["ens_val"], s["ens_test"])
if story_te is not None and ENS:
    b1000, b2000 = draw(story_te, 1000), draw(story_te, 2000)
    for setup, (ev, et) in ENS.items():
        thr, _ = pick_threshold(y_va, ev)
        valid = [b for b in b1000 if y_te[b].sum() > 0]
        for metric, fn in [("ROC_AUC", lambda b: roc_auc_score(y_te[b], et[b])), ("PR_AUC", lambda b: average_precision_score(y_te[b], et[b])),
                           ("F1", lambda b: f1_score(y_te[b], (et[b] >= thr).astype(int), zero_division=0))]:
            vals = np.array([fn(b) for b in valid]); full = fn(np.arange(len(y_te)))
            boot_rows.append(dict(model_pair=f"{setup} (ensemble of seeds 42/43/44, single model)", metric=metric, observed_difference=full, confidence_interval_lower=np.percentile(vals, 2.5), confidence_interval_upper=np.percentile(vals, 97.5),
                                  number_of_bootstrap_iterations=1000, random_seed=0, resampling_unit="story_id (whole stories, with replacement)", valid_resamples=len(valid), discarded_resamples=1000 - len(valid), source="RECOMPUTED_from_saved_probabilities (not a saved output); observed_difference column = the metric itself"))
    PAIRS = [("BASIL only", "Anno-lexical -> BASIL"), ("BASIL only", "BABE -> BASIL"), ("BASIL only", "Anno-lexical -> BABE -> BASIL"),
             ("BABE -> BASIL", "Anno-lexical -> BASIL"), ("Anno-lexical -> BASIL", "Anno-lexical -> BABE -> BASIL")]
    valid2 = [b for b in b2000 if y_te[b].sum() > 0]
    for a, b_ in PAIRS:
        if a not in ENS or b_ not in ENS: continue
        pa, pb = ENS[a][1], ENS[b_][1]
        for metric, fn in [("ROC_AUC", roc_auc_score), ("PR_AUC", average_precision_score)]:
            d = np.array([fn(y_te[i], pb[i]) - fn(y_te[i], pa[i]) for i in valid2]); obs = fn(y_te, pb) - fn(y_te, pa)
            boot_rows.append(dict(model_pair=f"{b_} minus {a}", metric=metric, observed_difference=obs, confidence_interval_lower=np.percentile(d, 2.5), confidence_interval_upper=np.percentile(d, 97.5),
                                  number_of_bootstrap_iterations=2000, random_seed=0, resampling_unit="story_id (whole stories, with replacement; same resamples for both setups)", valid_resamples=len(valid2), discarded_resamples=2000 - len(valid2), source="RECOMPUTED_from_saved_probabilities (the original paired_tests.py output was only printed, never saved)"))
if boot_rows:
    pd.DataFrame(boot_rows).to_csv(f"{OUT_DIR}/bootstrap_results.csv", index=False)
    inv("G", "bootstrap_results.csv", "PARTIAL", "intervals saved by the experiment (F1, PR-AUC in results.csv) are exported as such; the paired comparisons were only printed in Colab, so they are RECOMPUTED here from the saved probabilities with the same procedure (story resampling, seed 0, 2000 draws) and labelled as recomputed")
else:
    inv("G", "bootstrap_results.csv", "MISSING", "no saved bootstrap output and no probabilities to recompute from")

# ----------------------------- I) word level ---------------------------------------------
wl = []
for (folder, prefix), by_seed in runs.items():
    for seed, d in by_seed.items():
        if "hit" in d and "has" in d and "test_comb" in d:
            wmax = 2.0 * d["test_comb"].astype(np.float64) - d["test"].astype(np.float64)   # because test_comb = 0.5*(sentence + max word)
            wl.append(pd.DataFrame({"model_name": f"{folder}/{prefix}", "seed": seed, "row_index": te.row_index.values, "sentence_id": te.sentenceId.values, "article_id": te.articleId.values,
                                    "true_sentence_label": y_te, "sentence_probability": d["test"].astype(np.float64), "max_word_piece_probability_DERIVED": wmax,
                                    "top_piece_inside_gold_phrase": d["hit"].astype(int), "sentence_has_gold_phrase": d["has"].astype(int)}))
if wl:
    pd.concat(wl).to_csv(f"{OUT_DIR}/word_level_sentence_summary.csv", index=False)
inv("I", "word_level_predictions.csv", "MISSING", "token-level text, offsets and probabilities were never saved (the npz files keep only the sentence probability, the best word-piece probability and a hit flag per sentence). "
    + ("A sentence-level summary was exported instead as word_level_sentence_summary.csv (max_word_piece_probability is DERIVED from test_comb = 0.5*(sentence+word))." if wl else "No word-level run was found either."))

# ----------------------------- code / notebooks / logs -----------------------------------
src_files = []
for d_ in [f"{MYDRIVE}/Colab Notebooks", BASE, RUNS, EXTRA_DIR]:
    for pat in ["*.ipynb", "*.py", "*.log", "*.txt"]:
        src_files += glob.glob(f"{d_}/{pat}") + glob.glob(f"{d_}/*/{pat}")
src_files += [p for p in EXTRA_PATHS if os.path.exists(p)]
src_files = sorted({p for p in src_files if os.path.getsize(p) < 30_000_000 and "research_audit_export" not in p and "/stage1_anno_checkpoint" not in p and "_checkpoint" not in p})
KEYWORDS = ("AutoModelForSequenceClassification", "pick_threshold", "BASIL", "xlm-roberta")
code_cells, log_texts, skipped_nb = [], [], []
def nb_cells(path):
    try: nb = json.load(open(path, encoding="utf-8"))
    except Exception: return []
    cells = nb.get("cells") or sum([w.get("cells", []) for w in nb.get("worksheets", [])], [])
    out = []
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code": continue
        s = c.get("source", ""); s = "".join(s) if isinstance(s, list) else s
        txt = ""
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream": t = o.get("text", ""); txt += "".join(t) if isinstance(t, list) else t
            elif "data" in o and "text/plain" in o["data"]: t = o["data"]["text/plain"]; txt += "".join(t) if isinstance(t, list) else t
        out.append(dict(file=path, cell=i, exec=c.get("execution_count"), code=s, output=txt))
    return out
for p in src_files:
    if p.endswith(".ipynb"):
        cells = nb_cells(p)
        if any(any(k in c["code"] for k in KEYWORDS) for c in cells):
            code_cells += cells
            log_texts += [(p, c["cell"], c["output"]) for c in cells if c["output"]]
        else: skipped_nb.append(p)
    elif p.endswith(".py"):
        t = open(p, encoding="utf-8", errors="ignore").read()
        if any(k in t for k in KEYWORDS): code_cells.append(dict(file=p, cell=0, exec=None, code=t, output=""))
    else:
        t = open(p, encoding="utf-8", errors="ignore").read()
        if "epoch" in t and "train loss" in t: log_texts.append((p, 0, t))
try:                                    # code run in THIS runtime session, if the experiments were run here
    hist = get_ipython().history_manager.input_hist_raw
    for i, t in enumerate(hist):
        if any(k in t for k in KEYWORDS) and "research_audit_export" not in t: code_cells.append(dict(file="(current Colab session history)", cell=i, exec=None, code=t, output=""))
except Exception: pass

# K) experiment_source.py
if code_cells:
    with open(f"{OUT_DIR}/experiment_source.py", "w", encoding="utf-8") as fh:
        fh.write("# AUDIT COPY of the experiment code found in saved notebooks / scripts. It is READ-ONLY evidence, not meant to be run as one file.\n"
                 "# Notebooks are the SAVED versions: a cell may have been edited after it was executed. 'exec' is the cell's execution counter if the notebook stored one.\n\n")
        for c in code_cells:
            fh.write(f"\n# {'='*100}\n# SOURCE: {c['file'].replace(MYDRIVE, '<Drive>')} | cell {c['cell']} | exec {c['exec']}\n# {'='*100}\n{c['code']}\n")
    inv("K", "experiment_source.py", "EXPORTED", f"{len(code_cells)} code cells/files from {len({c['file'] for c in code_cells})} source(s); notebooks without experiment code were skipped: {len(skipped_nb)}")
else:
    inv("K", "experiment_source.py", "MISSING", "no saved notebook or .py with the experiment code was found (looked in 'Colab Notebooks', the BASIL_datasets folders and /content/extra_inputs). "
        "Upload the .ipynb files (File > Download > .ipynb from each Colab) to /content/extra_inputs and re-run.")

# D) training_history.csv  (parsed from the printed training logs; nothing else was ever saved)
pat = re.compile(r"\[(?P<tag>[^\]]+)\]\s+epoch\s+(?P<ep>\d+)/(?P<n>\d+)\s*\|\s*train loss\s+(?P<loss>[\d.]+)\s*\|\s*(?P<rest>.*)")
hist_rows = []
for p, cell, text in log_texts:
    for line in text.splitlines():
        m = pat.search(line)
        if not m: continue
        tag, rest = m["tag"], m["rest"]
        sm = re.search(r"seed (\d+)", tag)
        stage = ("stage1_anno_lexical" if "stage 1" in tag else "stage2_babe" if "BABE" in tag else "basil_finetune")
        hist_rows.append(dict(model_name=tag, seed=int(sm[1]) if sm else "", training_stage=stage, epoch=int(m["ep"]), epochs_planned=int(m["n"]),
                              training_loss=float(m["loss"]), validation_loss="NOT LOGGED",
                              validation_ROC_AUC=float(re.search(r"ROC-AUC\s+([\d.]+)", rest)[1]) if re.search(r"ROC-AUC\s+([\d.]+)", rest) else "NOT LOGGED",
                              validation_PR_AUC=float(re.search(r"PR-AUC\s+([\d.]+)", rest)[1]) if re.search(r"PR-AUC\s+([\d.]+)", rest) else "NOT LOGGED",
                              learning_rate="NOT LOGGED", validation_metric_note=rest.strip(), source_log=f"{p.replace(MYDRIVE, '<Drive>')} cell {cell}"))
if hist_rows:
    pd.DataFrame(hist_rows).to_csv(f"{OUT_DIR}/training_history.csv", index=False)
    inv("D", "training_history.csv", "PARTIAL", f"{len(hist_rows)} epoch lines parsed from saved Colab outputs (3-decimal values as printed). Validation loss and learning rate were never computed/logged. "
        "Re-run cells appear more than once (see source_log); nothing was de-duplicated.")
else:
    inv("D", "training_history.csv", "MISSING", "the training code only PRINTS per-epoch loss and validation metrics; no file was written. No saved notebook output or log containing those lines was found. "
        "Validation loss and learning rate were never logged at all.")

# A) experiment_config.json  (settings are read from the code that was found, never typed in from memory)
consts = {}
for c in code_cells:
    code = "\n".join(l for l in c["code"].splitlines() if not l.lstrip().startswith(("!", "%")))
    try: tree = ast.parse(code)
    except Exception: continue
    for node in tree.body:
        if isinstance(node, ast.Assign):
            tgts = node.targets[0]
            names = [t.id for t in tgts.elts] if isinstance(tgts, ast.Tuple) else [tgts.id] if isinstance(tgts, ast.Name) else []
            try: val = ast.literal_eval(node.value)
            except Exception: continue
            vals = list(val) if isinstance(tgts, ast.Tuple) and isinstance(val, (tuple, list)) else [val]
            if len(names) != len(vals): continue
            for n, v in zip(names, vals):
                if n.isupper() or n in ("MODES", "SEEDS"):
                    consts.setdefault(n, []).append(dict(value=v, source=c["file"].replace(MYDRIVE, "<Drive>"), cell=c["cell"], exec=c["exec"]))
FACTS = {   # a statement the report makes about the code  ->  regex that must be present in the code text
    "optimizer is AdamW": r"torch\.optim\.AdamW", "weight decay 0.01": r"weight_decay\s*=\s*0\.01",
    "linear schedule with warm-up": r"get_linear_schedule_with_warmup", "10% warm-up": r"int\(0\.1\s*\*",
    "gradient clipping 1.0": r"clip_grad_norm_\([^)]*,\s*1\.0\)", "mixed precision fp16": r"torch\.autocast\([^)]*float16",
    "unweighted cross-entropy for sentence loss": r"F\.cross_entropy\(\s*(ls|logits)\.float\(\),\s*labels\)",
    "word-level loss uses class weight": r"weight\s*=\s*wtok", "negative sampling per epoch (rng.choice of negatives)": r"rng\.choice\(neg_idx",
    "threshold = best validation F1 on grid 0.02..0.98": r"np\.linspace\(0\.02,\s*0\.98,\s*97\)", "ties keep lower threshold (strict >)": r"f1\s*>\s*best_f1\s*\+\s*1e-12",
    "epoch chosen by validation PR-AUC": r"average_precision_score\(y_va", "ensemble = mean of seed probabilities": r"np\.mean\(pvs,\s*0\)",
    "story-level bootstrap with default_rng(seed)": r"default_rng\(seed\)", "seed set (python/numpy/torch/cuda)": r"torch\.cuda\.manual_seed_all",
    "classifier head = AutoModelForSequenceClassification(num_labels=2)": r"AutoModelForSequenceClassification\.from_pretrained\([^)]*num_labels\s*=\s*2",
    "BABE 10% held-out, stratified, random_state=0": r"test_size\s*=\s*0\.1,\s*stratify=[^,]+,\s*random_state\s*=\s*0",
}
facts = {}
allcode = [(c["file"].replace(MYDRIVE, "<Drive>"), c["cell"], c["code"]) for c in code_cells]
for k, rx in FACTS.items():
    hits = [f"{f} cell {i}" for f, i, t in allcode if re.search(rx, t)]
    facts[k] = hits if hits else "NOT FOUND in any code located"
executed = [dict(run_folder=f, file_prefix=p, report_setup=KNOWN.get((f, p), "NOT IN REPORT"), seeds_with_saved_predictions=sorted(by), arrays_in_npz=sorted(next(iter(by.values())).keys()),
                 has_ensemble_prediction_file=os.path.exists(f"{RUNS}/{f}/{p}_ensemble_test_predictions.csv") or os.path.exists(f"{RUNS}/{f}/ensemble_test_predictions.csv"),
                 has_results_csv=os.path.exists(f"{RUNS}/{f}/results.csv")) for (f, p), by in sorted(runs.items())]
CLAIMED = {"stage1_anno_lexical": dict(epochs=2, lr=2e-5, batch=32, max_len=128), "stage2_babe": dict(epochs=3, lr=1e-5, batch=16, max_len=128), "stage3_basil": dict(epochs=8, lr=1e-5, batch=16, max_len=256)}
cfg = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), experiment_folder=BASE.replace(MYDRIVE, "<Drive>"),
           note="'constants_found_in_code' lists EVERY occurrence found in the saved code (several notebooks may differ); nothing was inferred. 'report_claims' are what the report says, for comparison only.",
           executed_runs_with_saved_predictions=executed, constants_found_in_code=consts if consts else "MISSING (no code located)",
           code_facts_checked=facts, report_claims=CLAIMED,
           selection_procedure_in_code={"epoch": "highest validation PR-AUC (average_precision_score on validation)", "threshold": "pick_threshold(): grid 0.02..0.98 step 0.01, best validation F1, ties keep the lower threshold",
                                        "ensemble": "arithmetic mean of the per-seed probabilities; ensemble threshold re-picked on the mean validation probabilities"} if allcode else "MISSING (code not located; see experiment_source.py status)",
           not_recorded_anywhere=["per-epoch validation loss", "learning rate per step/epoch", "library versions at the time the experiments ran", "token-level word-model predictions"])
def make_json_safe(obj):
    """Convert unsupported dictionary keys and values into JSON-safe forms."""
    if isinstance(obj, dict):
        return {
            str(k) if not isinstance(k, (str, int, float, bool)) and k is not None else k:
            make_json_safe(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

with open(f"{OUT_DIR}/experiment_config.json", "w", encoding="utf-8") as f:
    json.dump(make_json_safe(cfg), f, indent=2, default=str)
inv("A", "experiment_config.json", "EXPORTED" if consts else "PARTIAL", "settings come from the located code and saved run files (see 'constants_found_in_code' and 'code_facts_checked'); anything not found is marked, not filled in")

# J) environment.txt
def ver(mod):
    try: return __import__(mod).__version__
    except Exception: return "not installed"
gpu = "no GPU in this runtime"; cuda = "n/a"
try:
    import torch; cuda = str(torch.version.cuda); gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else gpu
except Exception: pass
try: smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
except Exception: smi = ""
env = ["NOTE: this describes the runtime that is running THIS audit script. Colab sessions are recycled, so it may differ from the runtime that trained the models. The experiments did not store their versions.",
       f"python: {sys.version.split()[0]}", f"torch: {ver('torch')}", f"transformers: {ver('transformers')}", f"tokenizers: {ver('tokenizers')}", f"scikit-learn: {ver('sklearn')}",
       f"numpy: {ver('numpy')}", f"pandas: {ver('pandas')}", f"CUDA (torch build): {cuda}", f"GPU: {gpu}", f"nvidia-smi: {smi or 'n/a'}", f"platform: {platform.platform()}"]
open(f"{OUT_DIR}/environment.txt", "w").write("\n".join(env) + "\n")
inv("J", "environment.txt", "PARTIAL", "versions of the CURRENT runtime only; the original training runtime's versions were not recorded")

# copy the experiments' own small result tables
os.makedirs(f"{OUT_DIR}/saved_results_csv", exist_ok=True)
for (folder, _) in runs:
    f = f"{RUNS}/{folder}/results.csv"; dst = f"{OUT_DIR}/saved_results_csv/{folder}_results.csv"
    if os.path.exists(f) and not os.path.exists(dst): shutil.copy(f, dst)

# ----------------------------- report vs saved results -----------------------------------
# numbers as printed in Research_Final_report_2 (Tables 4, 5, 6, confusion counts, 62%)
T4 = {"BASIL only": (.741, .174, .217, .192, .333), "BABE -> BASIL": (.804, .199, .251, .249, .254), "Anno-lexical -> BASIL": (.832, .212, .277, .244, .326),
      "Anno-lexical -> BABE -> BASIL": (.841, .208, .320, .224, .572), "Word-level (sentence head)": (.807, .180, .213, .190, .254)}
T5 = {"BASIL only": (.757, .175, .211), "BABE -> BASIL": (.815, .204, .249), "Anno-lexical -> BASIL": (.835, .213, .263), "Anno-lexical -> BABE -> BASIL": (.845, .212, .336), "Word-level (sentence head)": (.821, .184, .224)}
T6 = {("BASIL only", "Anno-lexical -> BASIL"): (.078, .035), ("BASIL only", "BABE -> BASIL"): (.058, .026), ("BASIL only", "Anno-lexical -> BABE -> BASIL"): (.088, .034),
      ("BABE -> BASIL", "Anno-lexical -> BASIL"): (.020, .009), ("Anno-lexical -> BASIL", "Anno-lexical -> BABE -> BASIL"): (.010, -.001)}
chk = []
def add_chk(item, claimed, found, tol=0.0006):
    if found is None: chk.append(dict(item=item, report=claimed, saved_outputs="MISSING", result="CANNOT CHECK")); return
    ok = abs(claimed - found) <= tol
    chk.append(dict(item=item, report=claimed, saved_outputs=round(float(found), 4), result="OK" if ok else "MISMATCH"))
if len(met):
    rec = met[(met.result_source.str.startswith("recomputed")) & (met.split == "test")]
    for setup in T4:
        sub = rec[(rec.report_setup == setup) & (~rec.model_name.str.endswith(":comb"))]
        per = sub[sub.seed.astype(str) != "ensemble"]; ens = sub[sub.seed.astype(str) == "ensemble"]
        three = sorted(per.seed.astype(int).tolist()) == SEEDS_EXPECTED
        for k, c in zip(["ROC_AUC", "PR_AUC", "F1", "precision", "recall"], T4[setup]): add_chk(f"Table 4 | {setup} | {k} (mean of 3 seeds)", c, per[k].mean() if three else None)
        for k, c in zip(["ROC_AUC", "PR_AUC", "F1"], T5[setup]): add_chk(f"Table 5 | {setup} | {k} (ensemble)", c, ens[k].iloc[0] if (three and len(ens)) else None)
    e3 = rec[(rec.report_setup == "Anno-lexical -> BABE -> BASIL") & (rec.seed.astype(str) == "ensemble")]
    for k, c in [("TP", 59), ("FP", 200), ("FN", 33), ("TN", 1255)]: add_chk(f"Section 5.1/5.3.5 | three-stage ensemble {k}", c, e3[k].iloc[0] if len(e3) else None, tol=0)
    for k, c in [("n_positive", 92), ("n", 1547)]: add_chk(f"Table 2 | test {k}", c, e3[k].iloc[0] if len(e3) else None, tol=0)
if boot_rows:
    bdf = pd.DataFrame(boot_rows)
    for (a, b_), (dr, dp) in T6.items():
        for metric, c in [("ROC_AUC", dr), ("PR_AUC", dp)]:
            s = bdf[(bdf.model_pair == f"{b_} minus {a}") & (bdf.metric == metric)]
            add_chk(f"Table 6 | {b_} minus {a} | {metric} difference", c, float(s.observed_difference.iloc[0]) if len(s) else None)
hits = []
for (folder, prefix), by_seed in runs.items():
    if prefix.startswith("span_anno_babe"):
        for seed, d in by_seed.items():
            if "hit" in d: hits.append(float(d["hit"][d["has"] & (y_te == 1)].mean()))
add_chk("Section 5.3.2 | top-piece hit rate (mean over seeds)", 0.62, float(np.mean(hits)) if len(hits) == 3 else None, tol=0.006)
checkdf = pd.DataFrame(chk); checkdf.to_csv(f"{OUT_DIR}/report_vs_saved_check.csv", index=False)
inv("extra", "report_vs_saved_check.csv", "EXPORTED", "every headline number of Tables 2, 4, 5, 6 and the confusion counts vs what the saved probabilities give (tolerance 0.0006)")
inv("extra", "saved_results_csv/", "EXPORTED", "copies of the experiments' own results.csv files (tiny)")
if ens_check: inv("extra", "ensemble_file_check.csv", "EXPORTED", "recomputed ensemble probabilities vs the ensemble_test_predictions.csv the experiment wrote")

# ----------------------------- L) inventory + zip ----------------------------------------
def size_of(fn):
    p = f"{OUT_DIR}/{fn}"
    if os.path.isdir(p): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
    return os.path.getsize(p) if os.path.exists(p) else 0
lines = ["RESEARCH AUDIT EXPORT - inventory", f"created: {datetime.datetime.now().isoformat(timespec='seconds')}", f"source experiment folder: {BASE.replace(MYDRIVE, '<Drive>')}", ""]
for r in INV:
    lines.append(f"[{r['item']}] {r['file']:<34} {r['status']:<9} {size_of(r['file'])/1024:9.1f} KB  {r['note']}")
if shape_problems: lines += ["", "PREDICTION FILES NOT EXPORTED (array length did not match the split file; nothing was guessed):"] + ["  " + s for s in shape_problems]
if skipped_nb: lines += ["", "Notebooks found but skipped (no experiment code in them): " + ", ".join(os.path.basename(s) for s in skipped_nb)]
open(f"{OUT_DIR}/audit_inventory.txt", "w").write("\n".join(lines) + "\n")
inv("L", "audit_inventory.txt", "EXPORTED", "")
zpath = shutil.make_archive("/content/research_audit_export" if not os.environ.get("AUDIT_OUT") else OUT_DIR + "_zip", "zip", os.path.dirname(OUT_DIR), os.path.basename(OUT_DIR))
zmb = os.path.getsize(zpath) / 1e6
if COPY_ZIP_TO_DRIVE and not BASE_OVERRIDE:
    os.makedirs(f"{BASE}/audit", exist_ok=True); shutil.copy(zpath, f"{BASE}/audit/research_audit_export.zip")

# ----------------------------- final printed report --------------------------------------
print("\n" + "=" * 90 + "\nAUDIT REPORT\n" + "=" * 90)
print("\n1) FILES")
for r in INV: print(f"   {r['status']:<9} [{r['item']}] {r['file']}")
print("\n2) MISSING / PARTIAL (and why)")
for r in INV:
    if r["status"] in ("MISSING", "PARTIAL"): print(f"   [{r['item']}] {r['file']}: {r['note']}")
print("\n3) PREDICTION COMPLETENESS (test + validation arrays, seeds found)")
for (folder, prefix), by in sorted(runs.items()):
    s = sorted(by); complete = s == SEEDS_EXPECTED
    print(f"   {'COMPLETE' if complete else 'INCOMPLETE':<10} {folder}/{prefix}  seeds={s}  -> {KNOWN.get((folder, prefix), 'not in report')}")
for setup in KNOWN.values():
    if not any(KNOWN.get(k) == setup for k in runs): print(f"   MISSING    no saved predictions for report setup: {setup}")
print("\n4) CAN THE REPORT'S METRICS BE REPRODUCED FROM WHAT WAS SAVED?")
if len(checkdf):
    n_ok = int((checkdf.result == "OK").sum()); n_bad = int((checkdf.result == "MISMATCH").sum()); n_na = int((checkdf.result == "CANNOT CHECK").sum())
    print(f"   {n_ok} numbers reproduced, {n_bad} MISMATCH, {n_na} cannot be checked (missing outputs)")
    for _, r in checkdf[checkdf.result != "OK"].iterrows(): print(f"   {r.result:<13} {r['item']}: report={r['report']} saved={r.saved_outputs}")
    verdict = "YES for the prediction-based metrics (Tables 2, 4, 5, 6, confusion counts)" if n_bad == 0 and n_na == 0 else "ONLY PARTLY - see the lines above"
else: verdict = "NO - no predictions were found"
print("   verdict:", verdict)
print("   Not reproducible from saved files in any case: per-epoch training curves (unless notebook outputs were found), validation loss,")
print("   learning-rate traces, token-level word-model predictions, the intermediate-stage ROC-AUC values (0.89 / 0.93 / 0.91), the shortcut-baseline")
print("   0.67, the 0.73/0.19/0.21 context experiment, and the original split-creation procedure (checked only as an outcome in split_manifest.csv).")
print(f"\nZIP: {zpath}  ({zmb:.2f} MB){'  <-- over 20 MB!' if zmb > 20 else ''}")
if COPY_ZIP_TO_DRIVE and not BASE_OVERRIDE: print(f"also copied to Drive: {BASE.replace(MYDRIVE, '<Drive>')}/audit/research_audit_export.zip")
try:
    from google.colab import files; files.download(zpath)
except Exception: pass

# ====================================================================================================
# SOURCE: (current Colab session history) | cell 1 | exec None
# ====================================================================================================
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
!ls /content/gdrive/MyDrive
!ls /content/gdrive/MyDrive/BASIL_datasets/corrected
