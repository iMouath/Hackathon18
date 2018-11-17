import tensorflow as tf
import os
from tensorflow import keras
import numpy as np
import keras.backend as K

def binary_pred(y_true, y_pred):
    return K.round(y_pred)

def decimalToBinary(dec, numberOfBits):
    answer = []
    while(dec > 0):
        answer.append(dec % 2)
        dec = dec // 2
    zeros = []
    for i in range(0, numberOfBits-len(answer)):
        zeros.append(0)
    return zeros + answer

def binaryToDecimal(bin):
    place = 1
    sum = 0
    for i in range(len(bin)-1, -1, -1):
        sum += (place * bin[i])
        place *= 2
    return sum

def genData(start, end):
    data = []
    labels = []
    for i in range(start, end+1):
        for j in range(0,11):
            num1Bin = decimalToBinary(i, 8)
            num2Bin = decimalToBinary(j, 8)
            answerBin = decimalToBinary(i * j, 16)
            data.append([num1Bin, num2Bin])
            labels.append(answerBin)
    return np.array(data), np.array(labels)

def generator(sequence_type):
    #training labels
    if sequence_type == 1:
        for i in range(6):
            for j in range(11):
                yield i * j
    #training data
    elif sequence_type == 2:
        for i in range(6):
            for j in range(11):
                yield (i, j)
    #testing labels
    elif sequence_type == 3:
        for i in range(6,11):
            for j in range(11):
                yield i * j
    #testing data
    elif sequence_type == 4:
        for i in range(6,11):
            for j in range(11):
                yield (i, j)

def displayDataset(dataset, numberOfElements):
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(numberOfElements):
            val = sess.run(next_element)
            print(val)

training_labels = tf.data.Dataset.from_generator(generator, (tf.int32), args = ([1]))
training_labels = training_labels.repeat(50)
#displayDataset(training_labels, 66*50)

training_data = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32), args = ([2]))
training_data = training_data.repeat(50)

testing_labels = tf.data.Dataset.from_generator(generator, (tf.int32), args = ([3]))
testing_labels = testing_labels.repeat(50)

testing_data = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32), args = ([4]))
testing_data = testing_data.repeat(50)

data, label = genData(1,5)
train_data_ds = tf.data.Dataset.from_tensor_slices(data)
train_label_ds = tf.data.Dataset.from_tensor_slices(label)

data2, label2 = genData(6,10)
test_data_ds = tf.data.Dataset.from_tensor_slices(data2)
test_label_ds = tf.data.Dataset.from_tensor_slices(label2)

train_data_ds = train_data_ds.repeat(1000)
train_label_ds = train_label_ds.repeat(1000)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2, 8)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy', binary_pred])

model.fit(data, label,  epochs = 10,
          validation_data = (data2, label2)) # pass callback to training
