import nltk
import matplotlib.pyplot as plt
import numpy as np

from nltk.corpus import brown


def corpus_statistics(corpus, title, categories=None):
    words = nltk.FreqDist(corpus.words(categories=categories))
    types = nltk.FreqDist([x[1] for x in corpus.tagged_words(categories=categories)])
    pos = nltk.FreqDist([x[1] for x in nltk.tag.pos_tag(brown.words(categories=categories))])
    print(title)
    print(f"\tNumber of unique tokens: {words.B()}")
    print(f"\tNumber of unique types: {types.B()}")
    print(f"\tTotal number of words: {words.N()}")
    print(f"\tAverage number of words per sentence: {words.N() / len(corpus.sents(categories=categories)):.2f}")
    print(f"\tAverage number of characters per word: {sum(map(len, corpus.words(categories=categories))) / words.N():.2f}")
    print(f"\tThe 10 most frequent POS tags: {pos.most_common(10)}")


def plot(fq, ax, title, lin_axes=True):
    freq_rank = np.asarray([(i, x[1]) for i, x in enumerate(fq.most_common())]).T
    ax.plot(*freq_rank if lin_axes else np.log(freq_rank))
    ax.set_xlabel("Rank" if lin_axes else "log(Rank)")
    ax.set_ylabel("Frequency" if lin_axes else "log(Frequency)")
    ax.set_title(title)


# Find word frequencies
fq = nltk.FreqDist(brown.words(categories=None))
fq_adv = nltk.FreqDist(brown.words(categories='adventure'))
fq_news = nltk.FreqDist(brown.words(categories='news'))

print(f"Full corpus word frequencies:\n\t{repr(fq)}")
print(f"Adventure category word frequencies:\n\t{repr(fq_adv)}")
print(f"News category word frequencies:\n\t{repr(fq_news)}")

# Print the required statistics
corpus_statistics(brown, "Adventure category statistics", categories='adventure')
print()
corpus_statistics(brown, "News category statistics", categories='news')

# Create a plot demonstrating Zipf's law
_, axs = plt.subplots(2, 2, figsize=(16, 12))
plot(fq_adv, axs[0][0], "Adventure genre", lin_axes=True)
plot(fq_news, axs[0][1], "News genre", lin_axes=True)
plot(fq_adv, axs[1][0], "Adventure genre (log)", lin_axes=False)
plot(fq_news, axs[1][1], "News genre (log)", lin_axes=False)
plt.suptitle("Token frequency plotted against rank")
plt.show()