#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

# TODO: read brown_vocab_100.txt into word_index_dict
with open("brown_vocab_100.txt", 'r') as f:
    word_index_dict = {w.rstrip(): i for i, w in enumerate(f.readlines())}

# TODO: write word_index_dict to word_to_index_100.txt
with open("word_to_index_100.txt", 'w') as f:
    f.write(str(word_index_dict))


print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
