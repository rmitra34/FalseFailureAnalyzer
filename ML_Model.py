"""Test result log analyser

This script allows the user to train or update the model on script execution logs.

This script requires that `sklean, imb-learn,numpy and pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * train_model - trains model from scratch
    * main - the main function of the script
    * update_model - updates an existing model
"""

import datetime
import os
from collections import Counter
import time

# external libraries
import joblib
import numpy as np
import seaborn as sn
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks


class Model:
    """ Encapsulates classifier related tasks and helps loading and saving model in one go """

    def __init__(self):
        self.time_stamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M")

        print('Model Stamp:' + self.time_stamp)

        self.clf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, criterion='gini',
                                          n_estimators=30, warm_start=True)

        self.vector = HashingVectorizer(n_features=2 ** 22, alternate_sign=False, analyzer='word',
                                        decode_error='ignore', token_pattern=r'\b\w{1,}[^\d\W]+\b',
                                        ngram_range=(2, 2))

        # Samplers are not needed during testing
        self.samplers = [
            TomekLinks(random_state=11, sampling_strategy='majority', n_jobs=-1),
            EditedNearestNeighbours(random_state=7, sampling_strategy='majority', n_jobs=-1)
        ]

    def load_model(self, time_stamp):
        """loads the model with provided timestamp
        Parameters:
        timestamp: time_stamp of the model to be loaded
        """
        file_path = os.path.join('models', time_stamp + '_modl.joblib')

        if not os.path.exists(file_path):
            print('No model file found with stamp: ' + time_stamp)
            return

        mdl = load_joblib(file_path)

        # Uses current time stamp for loaded model
        self.clf = mdl.clf
        self.vector = mdl.vector

        print('Model Loaded: ' + time_stamp)

    def fit_transform(self, text_train):
        """ Fit and transform text strings to frequency matrix.
        Parameters: array of training texts
        Returns: matrix of training data
        """

        add_to_log('Transforming..')
        s_time = time.time()

        x_train = self.vector.fit_transform(text_train)

        add_to_log('Transformation Time: ' + str(time.time() - s_time))

        # Save vector and matrix
        self.save_vector()
        self.save_matrix(x_train)

        return x_train

    def under_sample_data(self, matrix, y_train):

        """Remove samples from majority class to address bias in data.
         Reduces rows in x_train and y_train.

        Parameters:
        samples: x_train, labels: y_train

        Returns:
        updated samples: x_train, labels: y_train

       """

        add_to_log('Under Sampling')
        add_to_log('Original distribution %s' % Counter(y_train))
        s_time = time.time()

        x_res = matrix
        y_res = y_train

        for sampler in self.samplers:
            # clean proximity samples using TomeKLinks
            x_res, y_res = sampler.fit_resample(x_res, y_res)

        add_to_log('Adjusted distribution %s' % Counter(y_res))
        add_to_log('Under sampling time: ' + str(time.time() - s_time))

        return x_res, y_res

    def train_classifier(self, x_train, y_train):
        """Train and save a classifier.

        Parameters:
        samples: x_train, labels: y_train, classifier: RandomForest

        Returns:
        relative path to the classifier file inside models directory

       """
        add_to_log('Training Model..')
        s_time = time.time()

        self.clf.fit(x_train, y_train)

        add_to_log('Model Trained')
        add_to_log('Training Time: ' + str(time.time() - s_time))

        clf_path = self.save_classifier()

        return clf_path

    def update_classifier(self, x_train, y_train):
        """Update and save a classifier.

        Parameters:
        samples: x_train, labels: y_train, classifier: RandomForest

        Returns:
        relative path to the classifier file inside models directory

       """
        add_to_log('Training Model..')
        s_time = time.time()

        # Add a new decision tree per 300 samples
        new_estimators = len(y_train) // 300

        self.clf.n_estimators += new_estimators

        add_to_log('New estimators added: ' + str(new_estimators))

        self.clf.fit(x_train, y_train)

        add_to_log('Classifier Updated')
        add_to_log('Training Time: ' + str(time.time() - s_time))
        clf_path = self.save_classifier()

        return clf_path

    def get_predict_prob(self, text):
        """Return class conditional probability for a single sample
        Parameters: sample as text
        Returns: array of class probabilities
       """
        print('Vectorizing..')
        x_test = self.vector.fit_transform([text])
        y_preds = self.clf.predict_proba(x_test)
        print(y_preds[0])
        return y_preds[0]

    def score_accuracy(self, x_test, y_expec):
        """Scores accuracy of the model.
        Parameters: x_test, testing matrix, y_expec: expected labels
        """

        add_to_log('Scoring Model..')
        y_preds = self.clf.predict(x_test)

        acc = np.mean(y_preds == y_expec)
        add_to_log('accurary: ' + str(acc))

        self.print_confusion_matrix(y_expec, y_preds)

    def print_confusion_matrix(self, y_expec, y_preds):
        """Saves confusion matrix in png format.
        Parameters: Expected labels and Predicted labels.
        """

        clf_type = str(type(self.clf))
        clf_name = clf_type.split("'")[1].split('.')[-1]

        dir_path = os.path.join(os.getcwd(), 'cnf_mtrx')
        file_path = os.path.join(dir_path, self.time_stamp + clf_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        conf_mat = confusion_matrix(y_true=y_expec, y_pred=y_preds)

        conf_mat_pr = []
        for row in conf_mat:
            conf_mat_pr.append((row / sum(row)))

        add_to_log(conf_mat)
        acc = np.mean(y_preds == y_expec)

        # The order of labels is important
        labels = ['Hardware', 'Other', 'Script', 'Software', 'Tools']
        df_cm = DataFrame(conf_mat, index=labels, columns=labels)
        df_prec = DataFrame(conf_mat_pr, index=labels, columns=labels)

        sns_plot = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
        sns_plot.set_title("Acc: " + str(acc))
        plt.savefig(file_path)
        plt.figure()

        sns_plot = sn.heatmap(df_prec, annot=True, cmap='Blues', fmt='.2%')
        sns_plot.set_title("Acc: " + str(acc))
        plt.savefig(file_path + '_pr')
        plt.figure()

    def save_vector(self):
        """Saves vector inside the vectors folder.

        Parameters:
        model.time_stamp and vector object example: TF-IDF Vectorizer, Hashing Vectorizer etc.

        Returns:
        relative path to the file inside vectors directory

       """
        dir_path = os.path.join(os.getcwd(), 'vectors')
        file_path = os.path.join(dir_path, self.time_stamp + '_vctr.joblib')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        joblib.dump(self.vector, file_path)

        # Print vector attributes
        add_to_log('Vector Saved ' + file_path)
        if hasattr(self.vector, 'n_features'):
            add_to_log(self.vector.n_features)
        else:
            add_to_log(len(self.vector.get_feature_names()))

        add_to_log(self.vector.token_pattern)
        add_to_log(self.vector.ngram_range)

        if self.vector.stop_words is not None:
            add_to_log('Total Stop Words: ' + str(len(self.vector.stop_words)))
        else:
            add_to_log('No Stop Words')

        return file_path

    def save_matrix(self, x_train):
        """Saves matrix inside the matrix folder.s

        Parameters:
        model.time_stamp and matrix object example: X_train

       """
        dir_path = os.path.join(os.getcwd(), 'matrices')
        file_path = os.path.join(dir_path, self.time_stamp + '_mtrx.joblib')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        joblib.dump(x_train, file_path)
        add_to_log('Matrix Saved ' + file_path)

    def save_classifier(self):

        """Saves classifier inside the models folder.s

        Parameters:
        classifier: classifier object

        Returns:
        relative path to the file inside models directory

       """

        dir_path = os.path.join(os.getcwd(), 'classifiers')
        file_path = os.path.join(dir_path, self.time_stamp + '_clfr.joblib')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        joblib.dump(self.clf, file_path)

        add_to_log('Classifier Saved ' + file_path)

        return file_path

    def save_model(self):
        """ Saves model in self under model folder """

        dir_path = os.path.join(os.getcwd(), 'models')
        file_path = os.path.join(dir_path, self.time_stamp + '_modl.joblib')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        joblib.dump(self, file_path)
        add_to_log('Model Saved ' + file_path)


def load_joblib(file_path):

    """Loads the joblib file specified by file_path.

    Parameters:
    file_path (int): path to the file to load

    Returns:
    Object of the file, example: classifier, selector, vector

   """

    obj_file = joblib.load(file_path)
    add_to_log('File loaded ' + file_path)

    return obj_file


def add_to_log(line):

    """Appends the input to execution_log.txt file and prints as well.

        Parameters:
        line (string): String to be appended to log.
    """

    line = str(line)
    with open('execution_log.txt', 'a') as log:
        log.write(line)
        log.write('\n')
        if line == 'Done':
            log.write('-' * 50)
            log.write('\n')

    print(line)

