#!/usr/bin/env python
# coding: utf-8


token_corpus = []
for data in corpus:
    token_corpus.append(nltk.word_tokenize(data))

token_corpus[0]

import gensim
from gensim import models, similarities, corpora
import gensim.models.word2vec as w2v
import multiprocessing

num_features = 300
min_word_count = 1
num_workers = multiprocessing.cpu_count()
context_size = 3
downsampling = 1e-3
seed = 1

model = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling)

model.build_vocab(token_corpus)

print("Word2Vec vocabulary length:", len(model.wv.vocab))

model.train(token_corpus, len(model.wv.vocab), epochs=1)

model.save("word2vec.model")

model = w2v.Word2Vec.load("word2vec.model")

import sklearn
from sklearn import manifold

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = model.wv.syn0

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
        (word, all_word_vectors_matrix_2d[model.wv.vocab[word].index])
        for word in model.wv.vocab
    ]
    ],
    columns=["word", "x", "y"]
)

points.head(10)

import seaborn as sns

sns.set_context("poster")

points.plot.scatter("x", "y", s=10, figsize=(20, 12))


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


plot_region(x_bounds=(-400.0, -20), y_bounds=(-0.5, -0.1))

plot_region(x_bounds=(0, 0.2), y_bounds=(4, 4.5))

model.most_similar("clothing")

model.most_similar("mobile")

model.most_similar('furniture')

print(model)

similar_to_cloth = []
tmp_list = model.most_similar('clothing')
for data in tmp_list :
    similar_to_cloth.append(data[0])
similar_to_cloth.append('clothing')
similar_to_cloth.remove('t')

similar_to_cloth


for row in range(len(df)):
    cnt = False

    for data in similar_to_cloth:
        if data in str(df['breads'][row]):
            cnt = True

    if cnt == False:
        df['label'][row] = 'not clothing'
    else:
        df['label'][row] = 'clothing'

df.head()

# len(df)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm

#  Vectorizing our dataset
df.loc[df['label'] == 'clothing', 'label'] = 1
df.loc[df['label'] == 'not clothing', 'label'] = 0

df_x = df.breads
df_y = df.label

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=1)

cv = CountVectorizer()
# document-term-matrix
x_train_dtm = cv.fit_transform(x_train)
x_test_dtm = cv.transform(x_test)

# df.label.value_counts()
# pd.DataFrame(x_test_dtm.toarray(), columns=cv.get_feature_names())
# x_train_dtm

# Building and evaluating a model

# clf = svm.SVC()
# clf.fit(x_train_dtm, cv.transform(y_train).toarray())
# accuracy = clf.score(bdc_train, y_train)
# predict = clf.predict(bdc_test)
# print(accuracy)
