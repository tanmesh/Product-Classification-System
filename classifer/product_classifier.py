from gensim.models import Word2Vec
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

from generate_training_data import get_labelled_data


def do_word_embedding(input_data):
    model = Word2Vec(input_data, min_count=1)

    X_final = np.zeros((len(input_data), 50, 100))
    for i in range(len(input_data)):
        try:
            X_tmp = model[input_data[i]]
        except Exception:
            print(input_data[i])
        a = len(X_tmp)
        X_final[i, 0:a] = X_tmp
    return X_final


# THIS HAS EVERYTHING
def product_classifier():
    # get labelled data from csv file
    inputs, labels = get_labelled_data()
    print("labels generated successfully!")

    # word_embedding for raw input data
    inputs = do_word_embedding(inputs)
    print("data cleaned successfully!")

    # map data for SVM classifier
    inputs_array = np.asarray(inputs[:, :, -1])
    labels_array = np.asarray(labels)

    # splitting input training data
    train_input, test_input, train_labels, test_labels = train_test_split(inputs_array, labels_array)

    # running SVM
    classifier = svm.SVC(gamma='auto')
    classifier.fit(train_input, train_labels)

    # accuracy
    acc = classifier.score(test_input, test_labels)
    print("accuracy: " + str(acc))


# entry point
product_classifier()
