"""Collect fixed Top20 training candidates for 100 new, product-disjoint queries."""
import hashlib
import json
import os
from pathlib import Path
import torch
from retrieval import Retriever
from retrieval.rerank import CrossEncoderReranker, RerankerConfig

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/reranker_training_100'
PRODUCTS=[('Apple','iPhone 7'),('Apple','iPhone 7 Plus'),('Apple','iPhone SE'),
          ('Samsung','Galaxy S7'),('Samsung','Galaxy S7 Edge'),('Samsung','Galaxy S8'),
          ('Samsung','Galaxy S8 Plus'),('Samsung','Galaxy S9'),('Samsung','Galaxy S9 Plus'),
          ('Samsung','Galaxy Note 8'),('Samsung','Galaxy A51'),('Google','Pixel 3'),
          ('Google','Pixel 3 XL'),('Google','Pixel 3a'),('Google','Pixel 3a XL'),
          ('Motorola','Moto G Power (2022)')]
TEMPLATES=[
 ('camera','What do {name} owners report about camera quality, including blurry shots or photos in low light?'),
 ('battery','How does the battery of the {name} hold up during a full day, and do owners report unexpected drain?'),
 ('display','What do {name} reviews say about screen clarity, brightness, and touchscreen reliability?'),
 ('performance','How smoothly does the {name} run everyday apps, and what lag or freezing do owners experience?'),
 ('service','What happens when {name} buyers contact the seller or manufacturer about a faulty phone or a refund?'),
 ('carrier','What do {name} buyers report about SIM activation, unlocking, and compatibility with their carrier?')]

def queries():
    result=[]
    for brand,model in PRODUCTS:
        name=model if model.lower().startswith(brand.lower()) else brand+' '+model
        for aspect,template in TEMPLATES:
            result.append({'query_id':f'TR{len(result)+1:03d}','query':template.format(name=name),
                           'target_products':[{'brand':brand,'model':model}], 'aspects':[aspect],
                           'query_type':'single_aspect','product_group':brand+'/'+model})
    for index in [0,5,11,15]:
        brand,model=PRODUCTS[index]
        result.append({'query_id':f'TR{len(result)+1:03d}',
            'query':f'What do owners of the {brand} {model} say about both photo quality and battery life?',
            'target_products':[{'brand':brand,'model':model}], 'aspects':['camera','battery'],
            'query_type':'multi_aspect','product_group':brand+'/'+model})
    assert len(result)==100 and len({q['query'] for q in result})==100
    return result

def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(8)
    OUT.mkdir(exist_ok=True)
    qq=queries();qpath=OUT/'queries.jsonl'
    if qpath.exists():assert list(map(json.loads,qpath.read_text(encoding='utf8').splitlines()))==qq
    else:qpath.write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in qq),encoding='utf8',newline='\n')
    cache=OUT/'scored_candidates.jsonl'
    data={r['query_id']:r for r in map(json.loads,cache.read_text(encoding='utf8').splitlines())} if cache.exists() else {}
    if len(data)==100:
        assert set(data)=={q['query_id'] for q in qq}
        print('100 fixed candidate pools already available.');return
    retriever=Retriever.load(ROOT/'indexes')
    encoder=CrossEncoderReranker(RerankerConfig(local_files_only=True))._load_model()
    eval_models={p['model'] for r in map(json.loads,(ROOT/'evaluation/retrieval_ablation_40_v2/queries.jsonl').read_text(encoding='utf8').splitlines()) for p in r['target_products']}
    assert all(model not in eval_models for _,model in PRODUCTS)
    for q in qq:
        if q['query_id'] in data:continue
        res=retriever.retrieve(q['query'],mode='hybrid',k=50,candidate_k=150,strip_model=True,aspect_weight=.5,max_per_doc=1)
        rows=[h.to_dict() for h in res.hits[:20]]
        assert len(rows)==len({r['doc_id'] for r in rows})==20
        logits=[float(v) for v in encoder.predict([(res.search_text,r['text']) for r in rows],batch_size=16,show_progress_bar=False,convert_to_numpy=True)]
        data[q['query_id']]={'query_id':q['query_id'],'query':res.search_text,'product_group':q['product_group'],'rows':rows,'scores':logits}
        cache.write_text(''.join(json.dumps(data[k],ensure_ascii=False)+'\n' for k in sorted(data)),encoding='utf8',newline='\n')
        print(q['query_id'],'collected',flush=True)

if __name__=='__main__':main()
