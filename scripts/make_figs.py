import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import os, sys
# Usage: python make_figs.py <folder with test_predictions.csv & split_manifest.csv> <output folder>
D = "research_audit_export/research_audit_export/"   # folder where you unzipped the audit export
O = "figs/"                    # output folder for the figures
os.makedirs(O,exist_ok=True)
for f in ("test_predictions.csv","split_manifest.csv"):
    if not os.path.exists(D+f): raise SystemExit(f"Cannot find {D+f}. Unzip research_audit_export.zip first and pass its folder path.")
INK,INK2,GRID="#0b0b0b","#52514e","#e4e3df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,"text.color":INK,
 "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,
 "axes.axisbelow":True,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
tp=pd.read_csv(D+"test_predictions.csv"); sm=pd.read_csv(D+"split_manifest.csv"); tp["seed"]=tp.seed.astype(str)
SET=[("BASIL only","BASIL only","#2a78d6","o","-"),("BABE -> BASIL","BABE → BASIL","#eb6834","s","--"),
     ("Anno-lexical -> BASIL","Anno-lexical → BASIL","#1baf7a","^","-."),("Anno-lexical -> BABE -> BASIL","Anno-lexical → BABE → BASIL","#eda100","D",":")]
def get(s,ens=True):
    d=tp[(tp.report_setup==s)&((tp.seed=="ensemble") if ens else (tp.seed!="ensemble"))]
    return d.sort_values(["seed","row_index"])
E={s:get(s).reset_index(drop=True) for s,*_ in SET}
y=E["BASIL only"].true_label.values; prev=y.mean()
def save(fig,name):
    fig.savefig(O+name+".png",dpi=300,bbox_inches="tight"); fig.savefig(O+name+".pdf",bbox_inches="tight"); plt.close(fig)

# Fig 1 ROC + PR with operating points
fig,ax=plt.subplots(1,2,figsize=(7.4,3.5))
for s,lab,c,m,ls in SET:
    p=E[s].predicted_probability.values; thr=E[s].decision_threshold.iloc[0]
    fpr,tpr,_=roc_curve(y,p); pr,rc,_=precision_recall_curve(y,p)
    ax[0].plot(fpr,tpr,color=c,ls=ls,lw=1.8,label=f"{lab} (AUC {roc_auc_score(y,p):.3f})")
    ax[1].plot(rc,pr,color=c,ls=ls,lw=1.8,label=f"{lab} (AP {average_precision_score(y,p):.3f})")
    pred=p>=thr; P=(pred&(y==1)).sum()/max(pred.sum(),1); R=(pred&(y==1)).sum()/y.sum()
    ax[1].plot(R,P,marker=m,ms=8,mfc=c,mec="white",mew=1.5,ls="none")
    ax[0].plot((pred&(y==0)).sum()/(y==0).sum(),R,marker=m,ms=8,mfc=c,mec="white",mew=1.5,ls="none")
ax[0].plot([0,1],[0,1],color=INK2,lw=1,ls=(0,(1,3))); ax[1].axhline(prev,color=INK2,lw=1,ls=(0,(1,3)))
ax[1].text(0.02,prev-0.008,f"no-skill = prevalence {prev:.3f}",ha="left",va="top",fontsize=7.5,color=INK2)
ax[0].set(xlabel="False-positive rate",ylabel="True-positive rate (recall)",title="(a) ROC curves",xlim=(0,1),ylim=(0,1.01))
ax[1].set(xlabel="Recall",ylabel="Precision",title="(b) Precision–recall curves",xlim=(0,1),ylim=(0,0.6))
h,l=ax[0].get_legend_handles_labels(); ax[0].legend(h,l,loc="lower right",fontsize=7,frameon=False)
fig.text(0.5,-0.04,"Markers show the operating point at each setup's validation-selected threshold. Test set, 3-seed probability ensembles.",ha="center",fontsize=7.5,color=INK2)
fig.tight_layout(); save(fig,"fig_roc_pr")

# Fig 2 per-seed dots
fig,ax=plt.subplots(1,2,figsize=(7.4,3.2)); rows={}
for j,(met,f) in enumerate([("ROC-AUC",roc_auc_score),("PR-AUC",average_precision_score)]):
    for i,(s,lab,c,m,ls) in enumerate(SET):
        d=get(s,False); vals=[f(g.sort_values("row_index").true_label,g.sort_values("row_index").predicted_probability) for _,g in d.groupby("seed")]
        e=f(y,E[s].predicted_probability.values)
        ax[j].scatter(vals,[i]*3,s=46,color=c,marker=m,edgecolor="white",linewidth=1,zorder=5)
        ax[j].plot([min(vals),max(vals)],[i,i],color=c,lw=1.2,alpha=.6,zorder=2)
        ax[j].scatter([e],[i],s=150,facecolor="none",edgecolor=c,linewidth=1.6,marker="o",zorder=4)
    ax[j].set_yticks(range(len(SET))); ax[j].set_yticklabels([x[1] for x in SET] if j==0 else []); ax[j].invert_yaxis()
    ax[j].set_xlabel(met); ax[j].grid(axis="y",visible=False)
    if met=="PR-AUC": ax[j].axvline(prev,color=INK2,lw=1,ls=(0,(1,3))); ax[j].text(prev+.001,1.5,"no-skill",fontsize=7,color=INK2,va="center")
ax[0].set_title("(a) ROC-AUC per seed"); ax[1].set_title("(b) PR-AUC per seed")
fig.legend(handles=[Line2D([],[],marker="o",ls="none",color=INK2,ms=6,label="individual seed (42, 43, 44)"),Line2D([],[],marker="o",ls="none",mfc="none",mec=INK2,mew=1.6,ms=11,label="3-seed ensemble")],loc="lower center",ncol=2,fontsize=7.5,frameon=False,bbox_to_anchor=(0.55,-0.06))
fig.tight_layout(); save(fig,"fig_per_seed")

# Fig 3 forest of paired differences
story=E["BASIL only"].story_id.values; keys=pd.unique(story); idx={k:np.where(story==k)[0] for k in keys}
rng=np.random.default_rng(0); B=[np.concatenate([idx[k] for k in rng.choice(keys,len(keys))]) for _ in range(2000)]
B=[i for i in B if y[i].sum()>0]
pairs=[("BASIL only","BABE -> BASIL","BABE → BASIL  vs  BASIL only"),("BASIL only","Anno-lexical -> BASIL","Anno-lexical → BASIL  vs  BASIL only"),
 ("BASIL only","Anno-lexical -> BABE -> BASIL","Three-stage  vs  BASIL only"),("BABE -> BASIL","Anno-lexical -> BASIL","Anno-lexical → BASIL  vs  BABE → BASIL"),
 ("Anno-lexical -> BASIL","Anno-lexical -> BABE -> BASIL","Three-stage  vs  Anno-lexical → BASIL")]
fig,ax=plt.subplots(1,2,figsize=(7.4,2.9),sharey=True); res=[]
for j,(met,f) in enumerate([("ROC-AUC",roc_auc_score),("PR-AUC",average_precision_score)]):
    for i,(a,b,lab) in enumerate(pairs):
        pa,pb=E[a].predicted_probability.values,E[b].predicted_probability.values
        obs=f(y,pb)-f(y,pa); bs=[f(y[k],pb[k])-f(y[k],pa[k]) for k in B]; lo,hi=np.percentile(bs,[2.5,97.5]); res.append((met,lab,obs,lo,hi))
        sig=(lo>0 or hi<0); c="#2a78d6"
        ax[j].plot([lo,hi],[i,i],color=c,lw=2.2,solid_capstyle="round",alpha=1 if sig else .45)
        ax[j].plot([obs],[i],marker="o",ms=7,mfc=c if sig else "white",mec=c,mew=1.6,zorder=3)
        ax[j].text(hi+.004,i,f"{obs:+.3f}",va="center",fontsize=7,color=INK2)
    ax[j].axvline(0,color=INK,lw=1); ax[j].set_title(f"({'ab'[j]}) {met} difference"); ax[j].set_xlabel("Difference (95% interval)"); ax[j].grid(axis="y",visible=False)
    ax[j].set_xlim(-0.05,0.17 if j==0 else 0.13)
ax[0].set_yticks(range(5)); ax[0].set_yticklabels([p[2] for p in pairs]); ax[0].invert_yaxis()
fig.text(0.5,-0.05,"Intervals: 95% story-level bootstrap. Filled marker = interval excludes zero; open marker = includes zero. 2,000 paired resamples of test stories, seed 0; no multiple-comparison correction.",ha="center",fontsize=7.2,color=INK2)
fig.tight_layout(); save(fig,"fig_forest")
pd.DataFrame(res,columns=["metric","comparison","observed","ci_lo","ci_hi"]).round(4).to_csv(O+"forest_values.csv",index=False)

# Fig 4 story overlap
art=sm.drop_duplicates("article_id")[["article_id","story_id","split"]]
lay=art.groupby("story_id").split.value_counts().unstack(fill_value=0)[["train","validation","test"]]
cnt=lay.value_counts().rename("n").reset_index()
cnt["lab"]=cnt.apply(lambda r:f"{r.train}/{r.validation}/{r.test}",axis=1); cnt=cnt.sort_values("n",ascending=True)
te=sm[sm.split=="test"]; ov=te.story_id.isin(set(sm[sm.split=="train"].story_id))
fig,ax=plt.subplots(1,2,figsize=(7.4,3.1),gridspec_kw={"width_ratios":[1.15,1]})
cols=["#2a78d6" if (r.train>0 and (r.test>0 or r.validation>0)) else "#9a9992" for r in cnt.itertuples()]
b=ax[0].barh(cnt.lab,cnt.n,color=cols,height=.62)
for rect,v in zip(b,cnt.n): ax[0].text(v+.6,rect.get_y()+rect.get_height()/2,str(v),va="center",fontsize=8)
ax[0].set(xlabel="Number of stories",title="(a) Stories by split layout",xlim=(0,38)); ax[0].set_ylabel("Articles per story in train / val / test"); ax[0].grid(axis="y",visible=False)
ax[0].legend(handles=[plt.Rectangle((0,0),1,1,color="#9a9992"),plt.Rectangle((0,0),1,1,color="#2a78d6")],labels=["kept within one split","spread across splits"],loc="lower right",fontsize=7,frameon=False)
n_ov,n_dis=int(ov.sum()),int((~ov).sum()); p_ov,p_dis=int(te[ov].label.sum()),int(te[~ov].label.sum())
x=np.arange(2); ax[1].bar(x,[n_ov,n_dis],color=["#2a78d6","#9a9992"],width=.55)
for xi,(n,p) in zip(x,[(n_ov,p_ov),(n_dis,p_dis)]): ax[1].text(xi,n+25,f"{n:,} sentences\n{p} biased",ha="center",fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(["Test stories with a\ntraining article","Test stories with no\ntraining article"]); ax[1].set(ylim=(0,1750),title="(b) Test sentences by story overlap",ylabel="Test sentences"); ax[1].grid(axis="x",visible=False)
fig.tight_layout(); save(fig,"fig_story_overlap")

# Fig 5 confusion matrices
fig,ax=plt.subplots(1,2,figsize=(6.4,3.0))
for a,(s,lab) in zip(ax,[("BASIL only","BASIL only (threshold %.2f)"),("Anno-lexical -> BABE -> BASIL","Three-stage (threshold %.2f)")]):
    p=E[s].predicted_probability.values; thr=E[s].decision_threshold.iloc[0]; pr=p>=thr
    tn,fp,fn,tp_=((~pr)&(y==0)).sum(),(pr&(y==0)).sum(),((~pr)&(y==1)).sum(),(pr&(y==1)).sum()
    M=np.array([[tn,fp],[fn,tp_]]); R=M/M.sum(1,keepdims=True)
    a.imshow(R,cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b",["#ffffff","#256abf"]),vmin=0,vmax=1)
    for i in range(2):
        for j in range(2): a.text(j,i,f"{M[i,j]:,}\n({R[i,j]*100:.0f}% of row)",ha="center",va="center",fontsize=8.5,color="white" if R[i,j]>.5 else INK)
    a.set_xticks([0,1]); a.set_xticklabels(["Predicted\nnon-biased","Predicted\nbiased"]); a.set_yticks([0,1]); a.set_yticklabels(["Actual\nnon-biased","Actual\nbiased"]); a.grid(False)
    for sp in a.spines.values(): sp.set_visible(False)
    a.set_title(lab%thr,fontsize=8.5)
fig.tight_layout(); save(fig,"fig_confusion")
print(pd.DataFrame(res,columns=["m","c","o","lo","hi"]).round(3).to_string())