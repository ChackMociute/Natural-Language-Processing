#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE

#load the indices dictionary
with open("word_to_index_100.txt", 'r') as f:
    w2i = eval(f.read())

with open("brown_100.txt", 'r') as f:
    words = [w.lower() for s in f.readlines() for w in s.split()]

with open("toy_corpus.txt", 'r') as f:
    toy = [[w.lower() for w in s.split()] for s in f.readlines()]

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


sentprob = np.asarray([np.prod([
    probs[w2i[w1], w2i[w2]]
    for w1, w2 in zip(s[:-1], s[1:])])
    for s in toy
])
perplexity = 1/np.power(sentprob, 1/(np.asarray(list(map(len, toy))) - 1))
np.savetxt("bigram_eval.txt", perplexity)

with open("bigram_generation.txt", 'w') as f:
    for _ in range(10):
        f.write(GENERATE(w2i, probs, "bigram", 30, '<s>'))
        f.write('\n')