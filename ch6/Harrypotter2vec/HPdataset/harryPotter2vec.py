#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function


# In[2]:


import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re


# In[3]:


import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[4]:


# get_ipython().run_line_magic('pylab', 'inline')


# In[5]:


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[6]:


nltk.download("punkt")
nltk.download("stopwords")


# In[7]:


book_filenames = sorted(glob.glob("/home/wang/nlp_learn/NLPython-master/ch6/Harrypotter2vec/HPdataset/*.txt"))


# In[8]:


print("Found books:")
print(book_filenames)


# In[9]:


corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()


# In[10]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[11]:


raw_sentences = tokenizer.tokenize(corpus_raw)


# In[12]:


#convert into a list of words
#rtemove unnnecessary,, split into words, no hyphens
#list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[13]:


#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[14]:


print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))


# In[15]:


token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))


# In[16]:


#ONCE we have vectors
#step 3 - build model
#3 main tasks that vectors help with
#DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
#more dimensions, more computationally expensive to train
#but also more accurate
#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1


# In[17]:


harry2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[18]:


harry2vec.build_vocab(sentences)


# In[19]:


print("Word2Vec vocabulary length:", len(harry2vec.wv.vocab))


# In[21]:


harry2vec.train(sentences, total_examples=harry2vec.corpus_count, total_words=harry2vec.corpus_total_words
                  , epochs=harry2vec.epochs)


# In[22]:


if not os.path.exists("trained"):
    os.makedirs("trained")


# In[23]:


harry2vec.save(os.path.join("trained", "harry2vec.w2v"))


# In[24]:


harry2vec = w2v.Word2Vec.load(os.path.join("trained", "harry2vec.w2v"))


# In[25]:


tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)


# In[26]:


all_word_vectors_matrix = harry2vec.wv.syn0


# In[27]:


all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


# In[28]:


points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[harry2vec.wv.vocab[word].index])
            for word in harry2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)


# In[29]:


print(points.head(10))


# In[30]:


sns.set_context("poster")


# In[31]:


points.plot.scatter("x", "y", s=10, figsize=(20, 12))


# In[32]:


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


# In[38]:


plot_region(x_bounds=(4.0, 4.2), y_bounds=(-1.5, -0.1))


# In[39]:


plot_region(x_bounds=(0, 1), y_bounds=(1, 2.5))


# In[40]:


print(harry2vec.wv.most_similar("Harry"))


# In[41]:


print(harry2vec.most_similar("wand"))


# In[42]:


print(harry2vec.most_similar("Hogwarts"))


# In[43]:


def nearest_similarity_cosmul(start1, end1, end2):
    similarities = harry2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


# In[44]:


nearest_similarity_cosmul("Harry", "Potter", "Ron")
nearest_similarity_cosmul("Sirius", "Lupin", "Snape")
nearest_similarity_cosmul("hogwarts", "azkaban", "Phoenix")

plt.show()

# In[ ]:




