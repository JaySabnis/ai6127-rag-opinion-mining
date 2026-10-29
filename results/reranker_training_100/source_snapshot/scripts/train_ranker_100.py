"""Train only on the dedicated 100 queries; evaluate once on the separate 40."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from retrieval.learned_rank import FEATURES, features, predict
from scripts.design_reranker import metrics, order

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evaluation/reranker_training_100'
OUT=ROOT/'results/reranker_training_100'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=[json.loads(s) for s in (DATA/'scored_candidates.jsonl').read_text(encoding='utf8').splitlines()]
    qrels=[json.loads(s) for s in (DATA/'qrels.jsonl').read_text().splitlines()]
    labels={}
    for r in qrels:labels.setdefault(r['query_id'],{})[r['doc_id']]=r['relevance']
    assert len(data)==100 and len(qrels)==2000
    xs=[features(d['query'],d['rows'],d['scores']) for d in data]
    ys=[np.array([labels[d['query_id']][r['chunk_id']] for r in d['rows']],float) for d in data]
    groups=list(dict.fromkeys(d['product_group'] for d in data))
    folds=np.array([groups.index(d['product_group'])%5 for d in data])
    subsets=[[0,1,2],[0,1,2,3,4,10,11],[0,1,2,5,6,7,8],list(range(len(FEATURES))) ]
    grid=[(cols,alpha,target) for cols in subsets for alpha in [1,10,100,1000] for target in ['grade','binary']]
    def fit(indices,setting):
        cols,alpha,target=setting
        x=np.vstack([xs[i][:,cols] for i in indices])
        y=[]
        for i in indices:
            v=ys[i] if target=='grade' else (ys[i]>=2).astype(float)
            y.extend(v-v.mean())
        return np.linalg.solve(x.T@x+alpha*np.eye(len(cols)),x.T@np.array(y))
    def evaluate(indices,setting,beta):
        result=[]
        for i in indices:
            ranks=sorted(range(20),key=lambda j:float(xs[i][j,setting[0]]@beta),reverse=True)
            result.append(metrics([data[i]['rows'][j]['chunk_id'] for j in ranks],labels[data[i]['query_id']]))
        return np.array(result)
    cv=np.zeros((len(grid),len(data),3))
    for g,setting in enumerate(grid):
        for fold in range(5):
            train=np.flatnonzero(folds!=fold);valid=np.flatnonzero(folds==fold)
            cv[g,valid]=evaluate(valid,setting,fit(train,setting))
    best=int(np.argmax(cv[:,:,2].mean(axis=1)))
    nested=np.zeros((len(data),3));selected=[]
    for fold in range(5):
        train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
        scores=[]
        for setting in grid:
            values=[]
            for inner in range(5):
                if inner==fold:continue
                itrain=train[folds[train]!=inner];ival=train[folds[train]==inner]
                values.extend(evaluate(ival,setting,fit(itrain,setting))[:,2])
            scores.append(np.mean(values))
        chosen=int(np.argmax(scores));selected.append({'outer_fold':fold,'setting_index':chosen,'valid_queries':[data[i]['query_id'] for i in test]})
        nested[test]=evaluate(test,grid[chosen],fit(train,grid[chosen]))
    cols,alpha,target=grid[best];beta=fit(range(100),grid[best])
    model={'version':2,'columns':cols,'features':[FEATURES[i] for i in cols],
           'coefficients':beta.tolist(),'alpha':alpha,'target':target,'window':20,
           'training_queries':100,'training_pairs':2000,'training_dataset':'evaluation/reranker_training_100',
           'qrels_sha256':hashlib.sha256((DATA/'qrels.jsonl').read_bytes()).hexdigest(),
           'query_sha256':hashlib.sha256((DATA/'queries.jsonl').read_bytes()).hexdigest(),
           'selection':'five-fold product-group CV on training100 only; maximize nDCG@10',
           'evaluation40_used_for_training_or_selection':False}
    model_path=ROOT/'retrieval/models/review_ranker_v2.json'
    model_path.write_text(json.dumps(model,indent=2)+'\n',encoding='utf8')
    (OUT/'model.json').write_bytes(model_path.read_bytes())
    report={'training_queries':100,'training_pairs':2000,'training_products':len(groups),
            'metrics':['P@10','MRR@10','nDCG@10'],'model':model,
            'nested_product_group_cv':nested.mean(axis=0).tolist(),
            'selected_setting_group_cv':cv[best].mean(axis=0).tolist(),
            'training_fit':evaluate(range(100),grid[best],beta).mean(axis=0).tolist(),
            'folds':selected,'grid':[{'columns':c,'alpha':a,'target':t,'group_cv':cv[g].mean(axis=0).tolist()} for g,(c,a,t) in enumerate(grid)],
            'protocol':'All model fitting and selection uses training100 only. Outer and inner folds keep product groups together. MiniLM frozen. eval40 overlaps earlier model development, so it is held out from this fit, not an untouched final test.'}
    (OUT/'training_statistics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    with (OUT/'training_oof.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['query_id','product_group','fold','P@10','MRR@10','nDCG@10'])
        for i,d in enumerate(data):w.writerow([d['query_id'],d['product_group'],folds[i],*nested[i]])
    # Model artifact is frozen above before any eval40 labels are read.
    evaluate_frozen(model,OUT)
    print(json.dumps({'nested_product_group_cv':report['nested_product_group_cv'],
                      'selected':{'columns':cols,'alpha':alpha,'target':target},
                      'training_fit':report['training_fit']},indent=2))

def evaluate_frozen(model,out):
    eval_root=ROOT/'evaluation/retrieval_ablation_40_v2'
    labels={}
    for s in (eval_root/'qrels.jsonl').read_text().splitlines():
        r=json.loads(s);labels.setdefault(r['query_id'],{})[r['doc_id']]=r['relevance']
    extra_path=out/'evaluation_supplement.json'
    if extra_path.exists():
        for qid,items in json.loads(extra_path.read_text()).items():
            for cid,grade in items.items():
                assert cid not in labels[qid]
                labels[qid][cid]=grade
    (out/'evaluation_qrels.trec').write_text(''.join(
        f'{qid} 0 {cid} {g}\n' for qid,items in labels.items() for cid,g in sorted(items.items())),encoding='utf8')
    old=json.loads((ROOT/'retrieval/models/review_ranker_v1.json').read_text())
    systems=['retrieval','legacy','learned_v1','learned_v2']
    values={k:[] for k in systems};lines={k:[] for k in systems};per_query=[]
    for qid in sorted(labels):
        d=json.loads((ROOT/f'results/ablation_8/collection/{qid}.json').read_text(encoding='utf8'))
        baseline=sorted(d['runs']['08_learned'],key=lambda r:r['retrieval_rank'])
        head=baseline[:20];raw=[r['cross_encoder_raw_score'] for r in head]
        query=d['details']['08_learned']['scoring_query']
        v2=predict(query,head,raw,model);v1=predict(query,head,raw,old)
        rankings={'retrieval':baseline,'legacy':d['runs']['07_rewrite'],
                  'learned_v1':[head[i] for i in sorted(range(20),key=lambda i:v1[i],reverse=True)]+baseline[20:],
                  'learned_v2':[head[i] for i in sorted(range(20),key=lambda i:v2[i],reverse=True)]+baseline[20:]}
        for name,ranked in rankings.items():
            ids=[r['chunk_id'] for r in ranked];assert len(ids)==100
            assert all(cid in labels[qid] for cid in ids[:10])
            metric=metrics(ids,labels[qid]);nrel=sum(g>=2 for g in labels[qid].values())
            recall=sum(labels[qid].get(cid,0)>=2 for cid in ids)/nrel
            vals=[metric[0],recall,metric[1],metric[2]];values[name].append(vals)
            per_query.append([name,qid,*vals])
            lines[name].extend(f'{qid} Q0 {cid} {r} {101-r} {name}' for r,cid in enumerate(ids,1))
    means={name:np.mean(v,axis=0).tolist() for name,v in values.items()}
    delta=np.array(values['learned_v2'])-np.array(values['learned_v1'])
    rng=np.random.default_rng(42);boot=delta[rng.integers(0,40,(10000,40))].mean(axis=1)
    evaluation={'metrics':['P@10','pooled_Recall@100','MRR@10','nDCG@10'],
        'query_count':40,'qrels':'evaluation_qrels.trec: v2 qrels plus frozen-model Top10 supplement, shared by all four metrics',
        'supplemental_judgments':sum(len(items) for items in json.loads(extra_path.read_text()).values()) if extra_path.exists() else 0,
        'summary':means,'v2_minus_v1':{'mean':delta.mean(axis=0).tolist(),'paired_bootstrap_ci95':np.quantile(boot,[.025,.975],axis=0).tolist(),
        'ndcg_wins_ties_losses':[int((delta[:,3]>1e-10).sum()),int((abs(delta[:,3])<=1e-10).sum()),int((delta[:,3]<-1e-10).sum())]},
        'protocol':'Same cached rewritten Top100 as ablation8; fixed Top20 window; frozen MiniLM logits reused, no retrieval/API rerun. Weights selected without eval40. v1 used eval40 in training; v2 did not. eval40 is not historically untouched.'}
    (out/'evaluation40.json').write_text(json.dumps(evaluation,indent=2)+'\n',encoding='utf8')
    (out/'runs').mkdir(exist_ok=True)
    for name,rows in lines.items():(out/'runs'/f'{name}.trec').write_text('\n'.join(rows)+'\n',encoding='utf8')
    with (out/'evaluation40_per_query.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['system','query_id',*evaluation['metrics']]);w.writerows(per_query)
    print(json.dumps(evaluation,indent=2))

if __name__=='__main__':main()
