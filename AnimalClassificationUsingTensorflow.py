import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.cross_validation import train_test_split as split
# Getting the data

data=pd.read_csv('D:\\SEM5\\Kaggle\\Projects\\Zoo Animal Classification\\zoo.csv')

X=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values
y[:]=y[:]-1
y=keras.utils.to_categorical(y,num_classes=7)

X_train,X_test,y_train,y_test=split(X,y,test_size=0.25,random_state=1)

num_classes=7
ndim=X.shape[1]
learning_rate=0.01
epochs=500

n_hidden_1=30
n_hidden_2=20
n_hidden_3=10

def evaluation(X,y,weights,bias):
    layer_1=tf.add(tf.matmul(X,weights['h1']),bias['h1'])
    layer_1=tf.nn.sigmoid(layer_1)
        
    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),bias['h2'])
    layer_2=tf.nn.sigmoid(layer_2)
    
    layer_3=tf.add(tf.matmul(layer_2,weights['h3']),bias['h3'])
    layer_3=tf.nn.sigmoid(layer_3)
    
    out=tf.add(tf.matmul(layer_3,weights['output']),bias['output'])
    out=tf.nn.relu(out)
    
    return out




weights={
        'h1':tf.Variable(tf.truncated_normal([ndim,n_hidden_1])),
        'h2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
        'h3':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
        'output':tf.Variable(tf.truncated_normal([n_hidden_3,num_classes]))
        }

bias={
        'h1':tf.Variable(tf.truncated_normal([n_hidden_1])),
        'h2':tf.Variable(tf.truncated_normal([n_hidden_2])),
        'h3':tf.Variable(tf.truncated_normal([n_hidden_3])),
        'output':tf.Variable(tf.truncated_normal([num_classes]))
        }

x=tf.placeholder(tf.float32,shape=[None,ndim])
targets=tf.placeholder(tf.float32,shape=[None,num_classes])

y_hat=evaluation(x,targets,weights,bias)

cost_function=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat,labels=targets))
training_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
correct_prediction=tf.equal(y_hat,targets)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    accuracy_history=[]
    for i in range(epochs):
        sess.run(training_step,feed_dict={x:X_train,targets:y_train})
        accuracy_history.append(sess.run(accuracy,feed_dict={x:X_train,targets:y_train}))
    
    plt.plot(accuracy_history,'g-')
    plt.title('Training Accuracy')
    plt.show()
    
    #Testing Phase
    acc=sess.run(accuracy,feed_dict={x:X_test,targets:y_test})
    plt.plot(acc,'k-')
    plt.title('Testing Accuracy')
    plt.show()
    print(acc)

import somoclu

