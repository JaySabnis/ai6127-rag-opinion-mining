"""Independent ranking arithmetic used for training validation."""
import math

def metrics(ids,labels):
    grades=[labels[c] for c in ids[:10]]
    dcg=lambda a:sum(g/math.log2(i+2) for i,g in enumerate(a))
    ideal=dcg(sorted(labels.values(),reverse=True)[:10])
    return [sum(g>=2 for g in grades)/10,
            next((1/(i+1) for i,g in enumerate(grades) if g>=2),0),
            dcg(grades)/ideal if ideal else 0.0]
