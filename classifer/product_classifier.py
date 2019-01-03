import nltk
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import multiprocessing

from classifer.generate_training_data import get_labelled_data


def do_word_embedding(input_data):
    # print("model starts running....")
    # model = Word2Vec(input_data, min_count=1)
    # model.train(input_data, total_examples=len(input_data), epochs=1)
    # model.save("word2vec.model")
    # print("model finished.")
    model = Word2Vec.load("word2vec.model")

    X_final = np.zeros((len(input_data), 100, 100))
    for i in range(len(input_data)):
        vec = 0
        for data in input_data[i]:
            vec += model[data]
        X_final[i, :, :] = vec
    return X_final


# THIS HAS EVERYTHING
def product_classifier():
    # get labelled data from csv file
    input_df = get_labelled_data()
    print("Labels generated successfully!")

    # word_embedding for raw input data
    inputs = do_word_embedding(input_df.loc[:, 'bread'])
    print("data cleaned successfully!")

    # # map data for SVM classifier
    # inputs_array = np.asarray(inputs[:, :, -1])
    # labels_array = np.asarray(labels)
    #
    # # splitting input training data
    # train_input, test_input, train_labels, test_labels = train_test_split(inputs_array, labels_array)
    #
    # # running SVM
    # classifier = svm.SVC(gamma='auto')
    # classifier.fit(train_input, train_labels)
    #
    # # accuracy
    # acc = classifier.score(test_input, test_labels)
    # print("accuracy: " + str(acc))


product_classifier()