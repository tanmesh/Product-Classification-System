import nltk
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import multiprocessing

from classifer.generate_training_data import get_labelled_data


def do_word_embedding(input_data):
    # print('Extracting vocab....')
    # vocab = []
    # for row in range(len(input_data)):
    #     list1 = nltk.word_tokenize(input_data.iloc[row])
    #     for data in list1:
    #         vocab.append(data)
    #
    # print("model starts running....")
    # model = Word2Vec(vocab, min_count=1)
    # model.train(vocab, total_examples=len(vocab), epochs=1)
    # model.save("word2vec.model")
    # print("model finished.")

    model = Word2Vec.load("word2vec.model")
    list1 = nltk.word_tokenize(input_data.iloc[0])
    # print("For " + list[0] + "vector is" + model[list1[0]])
    print(list1[0])

    X_final = np.zeros((len(input_data), 50, 100))

    # X_tmp = model[input_data[0]]
    # for i in range(len(input_data)):
    #     try:
    #         X_tmp = model[input_data[i]]
    #     except Exception:
    #         print(input_data[i])
    #     column = len(X_tmp)
    #     X_final[i, 0:column] = X_tmp

    return X_final


# THIS HAS EVERYTHING
def product_classifier():
    # get labelled data from csv file
    # get_labelled_data()
    # print("Labels generated successfully!")

    input_df = pd.read_csv('training_input.csv')

    print(input_df['bread'][0])
    # word_embedding for raw input data
    # inputs = do_word_embedding(input_df.loc[:, 'bread'])
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
