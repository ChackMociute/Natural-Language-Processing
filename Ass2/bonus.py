import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE

#load the indices dictionary
with open("word_to_index_100.txt", 'r') as f:
    w2i = eval(f.read())

with open("brown_100.txt", 'r') as f:
    words = [w.lower() for s in f.readlines() for w in s.split()]

counts_pairs = np.zeros((len(w2i), len(w2i)))
counts_words = np.zeros(len(w2i))

prev_word_idx = None

for word in words:
    word_idx = w2i[word]
    counts_words[word_idx] += 1
    
    if prev_word_idx is not None:
        counts_pairs[prev_word_idx, word_idx] += 1
    
    prev_word_idx = word_idx
    
N = len(words)
pmi_pairs = np.zeros((len(w2i), len(w2i)))
for i in range(pmi_pairs.shape[0]):
    for j in range(pmi_pairs.shape[1]):
        p_w1_w2 = counts_pairs[i][j] * N
        if counts_words[i] < 10 or counts_words[j] < 10:
            # Ignore
            pmi_pairs[i][j] = 1
        elif p_w1_w2 > 0:
            pmi_pairs[i][j] = np.log2(p_w1_w2 / (counts_words[i] * counts_words[j]))
            
sorted_indices = np.argsort(pmi_pairs.flatten())
key_list = list(w2i.keys())
val_list = list(w2i.values())

row_indices, col_indices = np.unravel_index(sorted_indices[:20], pmi_pairs.shape)
pairs_lowest = []
for w1, w2 in zip(row_indices, col_indices):
    pair = (key_list[val_list.index(w1)], key_list[val_list.index(w2)])
    pairs_lowest.append(pair)
    
row_indices, col_indices = np.unravel_index(sorted_indices[-20:], pmi_pairs.shape)
pairs_highest = []
for w1, w2 in zip(row_indices, col_indices):
    pair = (key_list[val_list.index(w1)], key_list[val_list.index(w2)])
    pairs_highest.append(pair)

pairs_lowest.reverse()
print("-=-=- Lowest PMI values -=-=-")
for rank, pair in enumerate(pairs_lowest, start=1):
    print(f"Rank {rank}: {pair[0]}, {pair[1]}")

pairs_highest.reverse()
print("-=-=- Highest PMI values -=-=-")
for rank, pair in enumerate(pairs_highest, start=1):
    print(f"Rank {rank}: {pair[0]}, {pair[1]}")
