#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE


#load the indices dictionary
with open("word_to_index_100.txt", 'r') as f:
    w2i = eval(f.read())

with open("brown_100.txt", 'r') as f:
    words = [w.lower() for s in f.readlines() for w in s.split()]

with open("toy_corpus.txt", 'r') as f:
    toy = [[w.lower() for w in s.split()] for s in f.readlines()]

counts = np.zeros(len(w2i))
for w in words:
    counts[w2i[w]] += 1

probs = counts / counts.sum()

print(counts)
assert np.isclose(probs[0], 0.00040519)
assert np.isclose(probs[-1], 0.00364668)

np.savetxt("unigram_probs.txt", probs)


sentprob = np.asarray([np.prod([probs[w2i[w]] for w in s]) for s in toy])
perplexity = 1/np.power(sentprob, 1/np.asarray(list(map(len, toy))))
np.savetxt("unigram_eval.txt", perplexity)

with open("unigram_generation.txt", 'w') as f:
    for _ in range(10):
        f.write(GENERATE(w2i, probs, "unigram", 30, '<s>'))
        f.write('\n')