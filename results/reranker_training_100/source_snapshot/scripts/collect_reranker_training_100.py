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
    OUT.mkdir(exist_ok=True);(OUT/'collection').mkdir(exist_ok=True)
    qq=queries();qpath=OUT/'queries.jsonl'
    serialized=''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in qq)
    if qpath.exists():assert qpath.read_text(encoding='utf8')==serialized
    else:qpath.write_text(serialized,encoding='utf8')
    retriever=Retriever.load(ROOT/'indexes')
    scorer=CrossEncoderReranker(RerankerConfig(strategy='legacy',local_files_only=True))
    old_models={p['model'] for line in (ROOT/'evaluation/retrieval_ablation_40_v2/queries.jsonl').read_text(encoding='utf8').splitlines() for p in json.loads(line)['target_products']}
    assert all(model not in old_models for _,model in PRODUCTS)
    old_ids={json.loads(s)['chunk_id'] for s in (ROOT/'evaluation/retrieval_ablation_40_v2/corpus.jsonl').read_text(encoding='utf8').splitlines()}
    for q in qq:
        dest=OUT/'collection'/f"{q['query_id']}.json"
        if dest.exists():continue
        res=retriever.retrieve(q['query'],mode='hybrid',k=50,candidate_k=150,
                               strip_model=True,aspect_weight=.5,max_per_doc=1)
        hits=[h.to_dict() for h in res.hits[:20]]
        assert len(hits)==20 and len({h['doc_id'] for h in hits})==20
        assert res.filters and res.parsed.model==retriever.parser.resolve_model(q['target_products'][0]['brand'],q['target_products'][0]['model'])
        assert not ({h['chunk_id'] for h in hits}&old_ids)
        # Save all raw logits before annotation; MiniLM remains frozen.
        scores=scorer.rerank(res.search_text,hits,query_aspects=[])
        logits={h['chunk_id']:h['cross_encoder_raw_score'] for h in scores}
        payload={**q,'search_text':res.search_text,'filters':res.filters,
                 'parsed_aspects':res.parsed.aspects,'rows':hits,
                 'scores':[logits[h['chunk_id']] for h in hits]}
        dest.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
        print(q['query_id'],q['product_group'],q['aspects'],flush=True)
    print('100 fixed Top20 collections complete. No labels assigned by a model API.')

if __name__=='__main__':main()
