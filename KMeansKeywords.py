import sklearn
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer;
from sklearn.cluster import SpectralClustering;
import datetime
import joblib
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

from sklearn.metrics import accuracy_score

stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [word for word in tokenizer.tokenize(text.lower())]


def save_matrix(matrix):
    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y_%b_%d_%H_%M")
    file_path = os.path.join('matrix', time_stamp+'_mtrx.joblib')
    joblib.dump(matrix, file_path)

    print('Matrix Saved')

def save_vector(vector):
    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y_%b_%d_%H_%M")
    file_path = os.path.join('vectors', time_stamp+'_vctr.joblib')
    joblib.dump(vector, file_path)

    print('Vector Saved')

def preprocessing():
    categories = ['Tools', 'Hardware', 'Other', 'Script', 'Software'];

    docs_to_train = sklearn.datasets.load_files('/Users/rishabm/Desktop/MergeFileJTOrg/data1', description=None,
                                                categories=categories, load_content=True, encoding='utf-8');

    X_train, X_test, y_train, y_test = train_test_split(docs_to_train.data, docs_to_train.target, test_size=0.2);

    punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "%"]
    stop_words = text.ENGLISH_STOP_WORDS.union(punc)
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1,1));
    print('Training...');
    X_train_tfidf = vectorizer.fit_transform(X_train);
    counts = X_train_tfidf;
    transform = TfidfTransformer();
    X_train_tfidf = transform.fit_transform(X_train_tfidf);
    word_features = vectorizer.get_feature_names();
    print(word_features[500:525]);
    save_vector(vectorizer);
    save_matrix(X_train_tfidf);
    return [X_train_tfidf, word_features, y_train, vectorizer, counts];

def loadmatrix(text):
    matrix = joblib.load(text);
    return matrix;

def loadvector(text):
    vector = joblib.load(text);
    return vector;


def clustering():
    categories = ['pos', 'neg'];
    docs_to_train = sklearn.datasets.load_files('/Users/rishabm/Desktop/IMDB/data1', description=None,
                                                categories=categories, load_content=True, encoding='utf-8');

    X_train, X_test, y_train, y_test = train_test_split(docs_to_train.data, docs_to_train.target, test_size=0.2);
    davo = preprocessing();
    matrix = davo[0];
    vectorizer = davo[3];
    words = vectorizer.get_feature_names();
    idfvals = vectorizer.vocabulary_;
    spectral = SpectralClustering(n_clusters=2, assign_labels='kmeans', n_neighbors= 5, affinity='nearest_neighbors', n_init=5, n_jobs=1);
    spectral.fit_predict(matrix);
    counts = davo[4];
    countsum = counts.toarray().sum(axis=0);
    #common_words = kmean.cluster_centers_.argsort()[:, -1:-31:-1]
    #for num, centroid in enumerate(common_words):
        #print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
    #plt.scatter(matrix[:,0], matrix[:,1], s=50, cmap='viridis');
    tory = "Terms per cluster: " + '\n'
    for i in range(2):
        tory += "Cluster %d:" % i + '\n'
        T = matrix[spectral.labels_ == i].indices
        for ind in T:
            keyword = words[ind];
            tory += keyword + ': ' + str(countsum[idfvals[keyword]]) + '\n';
    with open('clusterwords3.txt', 'w+') as outfile:
        outfile.write(tory);

    #centers = kmean.cluster_centers_;
    vals = {};
    vals['cluster'] = spectral.labels_;
    vals['variety'] = y_train;
    mat = confusion_matrix(vals['variety'], vals['cluster']);
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=categories,
                yticklabels=[0,1]);
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    plt.show();

def main():
    oversampling();

def oversampling():
    categories = ['Tools', 'Hardware', 'Other', 'Script', 'Software'];

    docs_to_train = sklearn.datasets.load_files('/Users/rishabm/Desktop/MergeFileJTOrg/data1', description=None,
                                                categories=categories, load_content=True, encoding='utf-8');

    X_train, X_test, y_train, y_test = train_test_split(docs_to_train.data, docs_to_train.target, test_size=0.2);
    ogY = Counter(y_train);
    print('OLD SAMPLES: ');
    for key,value in ogY.items():
        print(key, value);
    matrix = loadmatrix('matrix/2019_Jun_21_14_44_mtrx.joblib');
    adasyn = ADASYN();
    matrix_resampled, y_resampled = adasyn.fit_resample(matrix, y_train);
    y_resampled.astype(int);
    newY = Counter(y_resampled);
    print('NEW SAMPLES: ');
    for key,value in newY.items():
        print(key,value)





main();