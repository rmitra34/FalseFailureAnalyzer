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

import sys

# external libraries
import numpy as np

from sklearn.datasets import load_files
from ML_Model import Model


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


def get_training_data(path):

    """Loads the training data specified by path.

    Parameters:
    path (int): path to the dir to load, directory must have buckets as sub directories

    Returns:
    Array of text in training samples and class labels

   """

    logs_train = load_files(path)
    text_train, y_train = logs_train.data, logs_train.target
    add_to_log('Got Training Data')
    print('Classes', np.unique(y_train))

    return text_train, y_train


def train_model(train_data_path):
    """Trains the model from scratch.

    Parameters:
    path: Training data path. The target directory must have samples separated in class buckets.

   """
    add_to_log(train_data_path + ' Training')

    text_train, y_train = get_training_data(train_data_path)

    model = Model()

    # Transform the text into nd array
    x_train = model.fit_transform(text_train)

    # Data Synthesis
    x_train, y_train = model.under_sample_data(x_train, y_train)

    model.train_classifier(x_train, y_train)

    # Save the trained model as a joblib file
    model.save_model()


def update_model(time_stamp, train_data_path):

    """Updates an existing model.

    Parameters:
    time_stamp: time_stamp of the model example: 2019_Jul_19_12_13
    path: Training data path. The source directory must have samples separated in class buckets.

   """

    add_to_log(train_data_path + ' Training')

    text_train, y_train = get_training_data(train_data_path)

    model = Model()
    model.load_model(time_stamp)
    # Transform the text into nd array
    x_train = model.fit_transform(text_train)

    # Test the accuracy of current month on previously trained model.
    test_model(x_train, y_train, model)

    # Data Synthesis
    x_train, y_train = model.under_sample_data(x_train, y_train)

    model.update_classifier(x_train, y_train)

    # Save the trained model as a joblib file
    model.save_model()


def test_model(x_test, y_test, model):
    """Updates an existing model.

    Parameters:
    time_stamp: time_stamp of the model
    path: Training data path. The source directory must have samples separated in class buckets.

   """
    # Feature Selection
    # x_test = model.selector_transform(x_test)

    model.score_accuracy(x_test, y_test)
    add_to_log('Done')


def main():
    """Trains or Updates the model depending on attribute supplied.

    Parameters:
    path: Training data path. The target directory must have samples separated in class buckets.
    task: -t for training -u to update existing model.
    time stamp: time stamp to the model to be updated when using -u.
   """

    if len(sys.argv) > 1 and len(sys.argv) >= 3:

        task = sys.argv[1]
        data_path = sys.argv[2]

        if task == '-t':
            train_model(data_path)
        elif task == '-u':
            if len(sys.argv) == 4:
                time_stamp = sys.argv[3]
                update_model(time_stamp, data_path)
            else:
                print('Takes 3 arguments: task to perform, '
                      'data path and time stamp in case of update')

        else:
            print('Argument not recognized, use -t to train'
                  ' and -u to update.')

    else:
        print('Takes 2 or 3 arguments: task to perform and data path respectively'
              ',follow by time stamp in case of update')

    print('Done')


if __name__ == '__main__':
    main()
