import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def read_dataset():
    df = pd.read_csv('bank_note_data.csv')
    X = df[df.columns[0:4]].values
    y = df[df.columns[4]]

    encoder = LabelEncoder()
    encoder.fit(y)

    y = encoder.transform(y)
    Y = one_hot_encode(y)

    return (X, Y)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))

    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1

    return one_hot_encode

X, Y = read_dataset()

X, Y = shuffle(X, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=1)

learning_rate = 0.3
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
n_class = 2
model_path = 'BankNote'

n_hidden_1 = 10
n_hidden_2 = 10
n_hidden_3 = 10
n_hidden_4 = 10

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.flloat32, [None, n_class])

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    
