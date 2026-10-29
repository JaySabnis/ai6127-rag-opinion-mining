"""Query-local ranking features; never reads evaluation labels at inference."""
import math
import re
import numpy as np
from chunking.metadata import ASPECT_KEYWORDS, detect_aspects

FEATURES = ['semantic_raw','semantic_rank','retrieval_rank','dense','bm25',
            'aspect_coverage','keyword_count','keyword_density','length',
            'semantic_squared','dense_rank','bm25_rank']

def features(query, rows, raw):
    aspects=detect_aspects(query)
    words=set(w for a in aspects for w in ASPECT_KEYWORDS[a])
    ranks={i:r for r,i in enumerate(sorted(range(len(raw)),key=lambda i:raw[i],reverse=True),1)}
    result=[]
    for i,r in enumerate(rows):
        meta={**(r.get('metadata') or {}),**r}
        text=r['text'].lower()
        count=sum(len(re.findall(r'(?<!\w)'+re.escape(w)+r'(?!\w)',text)) for w in words)
        length=max(1,len(text.split()))
        coverage=len(set(meta.get('aspect_tags') or [])&set(aspects))/max(1,len(aspects))
        result.append([raw[i],1/(10+ranks[i]),1/(10+i+1),r.get('dense_score') or 0,
                       math.log1p(max(0,r.get('bm25_score') or 0)),coverage,
                       math.log1p(count),count/length,math.log1p(length),
                       raw[i]**2,1/(10+(r.get('dense_rank') or 1000)),
                       1/(10+(r.get('bm25_rank') or 1000))])
    x=np.asarray(result,dtype=float)
    # Per-query scaling makes different retrieval score scales comparable.
    return (x-x.mean(axis=0))/np.maximum(x.std(axis=0),1e-6)

def predict(query,rows,raw,model):
    x=features(query,rows,raw)
    return x[:,model['columns']]@np.asarray(model['coefficients'])
