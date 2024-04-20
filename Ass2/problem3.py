#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize

#load the indices dictionary
with open("word_to_index_100.txt", 'r') as f:
    w2i = eval(f.read())

with open("brown_100.txt", 'r') as f:
    words = [w.lower() for s in f.readlines() for w in s.split()]

counts = np.zeros((len(w2i), len(w2i)))

for w1, w2 in zip(words[:-1], words[1:]):
    counts[w2i[w1], w2i[w2]] += 1

probs = normalize(counts, norm='l1', axis=1)


assert probs[w2i['all'], w2i['the']] == 1
assert np.isclose(probs[w2i['the'], w2i['jury']], 0.08333333333333)

np.savetxt("bigram_probs.txt", np.asarray([
    probs[w2i['all'], w2i['the']],
    probs[w2i['the'], w2i['jury']],
    probs[w2i['the'], w2i['campaign']],
    probs[w2i['anonymous'], w2i['calls']]
]))