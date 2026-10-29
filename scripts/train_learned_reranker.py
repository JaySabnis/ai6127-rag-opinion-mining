"""Nested query-fold development of a small regularized learning-to-rank model."""
import json
import csv
from pathlib import Path
import hashlib
import numpy as np
from retrieval.learned_rank import FEATURES, features
from scripts.design_reranker import metrics, order

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/reranker_learned'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=json.loads((ROOT/'results/reranker_design/scores.json').read_text(encoding='utf8'))
    qids=sorted(data)
    labels={}
    for s in (ROOT/'evaluation/retrieval_ablation_40/qrels.jsonl').read_text().splitlines():
        r=json.loads(s);labels.setdefault(r['query_id'],{})[r['doc_id']]=r['relevance']
    xs=[features(data[q]['query'],data[q]['rows'],data[q]['scores']) for q in qids]
    ys=[np.array([labels[q][r['chunk_id']] for r in data[q]['rows']],float) for q in qids]
    groups=[[0,1,2],[0,1,2,3,4,10,11],[0,1,2,5,6,7,8],list(range(len(FEATURES))) ]
    grid=[(cols,alpha,target) for cols in groups for alpha in [1,10,100,1000] for target in ['grade','binary']]
    def fit(indices,setting):
        cols,alpha,target=setting
        xx=[]; yy=[]
        for i in indices:
            y=ys[i] if target=='grade' else (ys[i]>=2).astype(float)
            xx.append(xs[i][:,cols]); yy.append(y-y.mean())
        x=np.vstack(xx);y=np.concatenate(yy)
        return np.linalg.solve(x.T@x+alpha*np.eye(len(cols)),x.T@y)
    def evaluate(indices,setting,beta):
        cols=setting[0];out=[]
        for i in indices:
            indices_rank=sorted(range(len(xs[i])),key=lambda j:float(xs[i][j,cols]@beta),reverse=True)
            out.append(metrics([data[qids[i]]['rows'][j]['chunk_id'] for j in indices_rank],labels[qids[i]]))
        return np.array(out)
    folds=np.arange(len(qids))%5
    # Validation predictions for every setting, trained without its query fold.
    cv=np.zeros((len(grid),len(qids),3))
    for g,setting in enumerate(grid):
        for f in range(5):
            train=np.flatnonzero(folds!=f);test=np.flatnonzero(folds==f)
            cv[g,test]=evaluate(test,setting,fit(train,setting))
    best=int(np.argmax(cv[:,:,2].mean(axis=1)))
    # Outer folds select settings using inner training folds only.
    nested=np.zeros((len(qids),3)); selected=[]
    for f in range(5):
        train=np.flatnonzero(folds!=f); test=np.flatnonzero(folds==f)
        inner_scores=[]
        for setting in grid:
            vv=[]
            for inner in range(5):
                if inner==f:continue
                itrain=train[folds[train]!=inner];ival=train[folds[train]==inner]
                vv.extend(evaluate(ival,setting,fit(itrain,setting))[:,2])
            inner_scores.append(np.mean(vv))
        chosen=int(np.argmax(inner_scores));selected.append(chosen)
        nested[test]=evaluate(test,grid[chosen],fit(train,grid[chosen]))
    previous=np.array([metrics([data[q]['rows'][j]['chunk_id'] for j in order(data[q]['rows'],data[q]['scores'],.8,10)],labels[q]) for q in qids])
    beta=fit(range(len(qids)),grid[best]);cols,alpha,target=grid[best]
    model={'version':1,'columns':cols,'features':[FEATURES[i] for i in cols],
           'coefficients':beta.tolist(),'alpha':alpha,'target':target,'window':20,
           'training_queries':len(qids),'qrels_sha256':hashlib.sha256((ROOT/'evaluation/retrieval_ablation_40/qrels.jsonl').read_bytes()).hexdigest()}
    report={'metrics':['P@10','MRR@10','nDCG@10'],'previous':previous.mean(axis=0).tolist(),
            'nested_cv':nested.mean(axis=0).tolist(),'selected_setting_cv':cv[best].mean(axis=0).tolist(),
            'training_fit':evaluate(range(len(qids)),grid[best],beta).mean(axis=0).tolist(),
            'outer_selected_grid_indices':selected,'deployment_model':model,
            'protocol':'Fixed Top20; nested 5-fold query validation for model/regularization/target selection. Existing development queries, not untouched test data. Recall@100 not recomputed because Top100 is unavailable.'}
    delta=nested-previous
    rng=np.random.default_rng(42)
    boot=delta[rng.integers(0,len(qids),(10000,len(qids)))].mean(axis=1)
    report['nested_minus_previous']={'mean':delta.mean(axis=0).tolist(),
        'paired_bootstrap_ci95':np.quantile(boot,[.025,.975],axis=0).tolist(),
        'ndcg_wins_ties_losses':[int((delta[:,2]>1e-10).sum()),int((abs(delta[:,2])<=1e-10).sum()),int((delta[:,2]<-1e-10).sum())]}
    (OUT/'model.json').write_text(json.dumps(model,indent=2)+'\n',encoding='utf8')
    (OUT/'statistics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    with (OUT/'per_query.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['query_id','old_P@10','old_MRR@10','old_nDCG@10','nested_P@10','nested_MRR@10','nested_nDCG@10'])
        for i,q in enumerate(qids):w.writerow([q,*previous[i],*nested[i]])
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
