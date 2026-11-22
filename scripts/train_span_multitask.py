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