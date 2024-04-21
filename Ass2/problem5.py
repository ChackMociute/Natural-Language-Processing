import re
import numpy as np
from sklearn.preprocessing import normalize

#load the indices dictionary
with open("word_to_index_100.txt", 'r') as f:
    w2i = eval(f.read())

with open("brown_100.txt", 'r') as f:
    words = [w.lower() for s in f.readlines() for w in s.split()]

def trigram_prob(c1, c2, w, smooth=False, a=0.1):
    w_c = len(re.findall(f" {c1} {c2} {w} ", ' '.join(words)))
    c_c = len(re.findall(f" {c1} {c2} ", ' '.join(words)))
    if smooth:
        print(f"Smoothed p({w}|{c1}, {c2})={(w_c + a) / (c_c + len(w2i) * a):.4f}")
    else:
        print(f"Unsmoothed p({w}|{c1}, {c2})={w_c / c_c:.4f}")
        return w_c / c_c

trigram_prob('in', 'the', 'past', smooth=False)
trigram_prob('in', 'the', 'time', smooth=False)
trigram_prob('the', 'jury', 'said', smooth=False)
trigram_prob('the', 'jury', 'recommended', smooth=False)
trigram_prob('jury', 'said', 'that', smooth=False)
trigram_prob('agriculture', 'teacher', ',', smooth=False)
print()
trigram_prob('in', 'the', 'past', smooth=True)
trigram_prob('in', 'the', 'time', smooth=True)
trigram_prob('the', 'jury', 'said', smooth=True)
trigram_prob('the', 'jury', 'recommended', smooth=True)
trigram_prob('jury', 'said', 'that', smooth=True)
trigram_prob('agriculture', 'teacher', ',', smooth=True)