import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


ignore_number=False
ignore_request=False
ignore_ip=False
ignore_fail=False

split_by_time=True


LOG_DIR ="log"
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)


time_steps=60



learning_rate=0.001
batch_size=128
# Beta for L2 regularization
beta = 0.01

n_input=4
if ignore_number:
    n_input-=1
if ignore_request:
    n_input-=1
if ignore_ip:
    n_input-=1
if ignore_fail:
    n_input-=1

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[batch_size,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[batch_size])


def weights_and_biases(a, b, name):
    w = tf.get_variable("W_"+name, shape=[a, b],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b_"+name, shape=[b],initializer=tf.zeros_initializer())
    #b = tf.Variable(tf.zeros([b]))
    return w, b


weights_1, biases_1 = weights_and_biases(time_steps*n_input, 32,"fc1")
weights_2, biases_2 = weights_and_biases(32, 8,"fc2")
weights_3, biases_3 = weights_and_biases(32, 1,"fc3")

data=tf.reshape(x, [batch_size,time_steps*n_input])

fc1= tf.nn.relu(tf.matmul(data, weights_1) + biases_1)
#fc2= tf.nn.relu(tf.matmul(fc1, weights_2) + biases_2)
fc3= tf.matmul(fc1, weights_3) + biases_3

prediction=tf.squeeze(fc3)

#loss_function
reg = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) 

loss=tf.reduce_mean(tf.square(y-prediction))
loss = loss + reg * beta

opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

tf.summary.scalar('loss', loss)


suf=""
if split_by_time:
    suf="_time"

x_train=np.load('x_train'+suf+'.npy')
x_test=np.load('x_test'+suf+'.npy')
y_train=np.squeeze(np.load('y_train'+suf+'.npy'))
y_test=np.squeeze(np.load('y_test'+suf+'.npy'))

if ignore_number:
    x_train=np.delete(x_train, 3, 2)
    x_test=np.delete(x_test, 3, 2)
if ignore_request:
    x_train=np.delete(x_train, 2, 2)
    x_test=np.delete(x_test, 2, 2)
if ignore_ip:
    x_train=np.delete(x_train, 1, 2)
    x_test=np.delete(x_test, 1, 2)
if ignore_fail:
    x_train=np.delete(x_train, 0, 2)
    x_test=np.delete(x_test, 0, 2)


def getTrainBatch(iter):
    start=(iter*batch_size)%len(x_train)
    stop=((iter+1)*batch_size)%len(x_train)
    #print(start,stop)
    if stop<start:
        return np.concatenate((x_train[start:,:,:],x_train[:stop,:,:]), axis=0),np.concatenate((y_train[start:],y_train[:stop]), axis=0)
    else:
        return x_train[start:stop,:,:],y_train[start:stop]

def getTestBatch(iter):
    start=(iter*batch_size)%len(x_test)
    stop=((iter+1)*batch_size)%len(x_test)
    if stop<start:
        return np.concatenate((x_test[start:,:,:],x_test[:stop,:,:]), axis=0),np.concatenate((y_test[start:],y_test[:stop]), axis=0)
    else:
        return x_test[start:stop,:,:],y_test[start:stop]

#initialize variables
init=tf.global_variables_initializer()

sess = tf.Session()

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

with sess:
    sess.run(init)
    iter=1
    while iter<100000:
        batch_x,batch_y=getTrainBatch(iter)


        summary, _=sess.run([merged,opt], feed_dict={x: batch_x, y: batch_y})
        train_writer.add_summary(summary, iter)

        batch_x_test,batch_y_test = getTestBatch(iter)
        summary,los,pred=sess.run([merged,loss,prediction],feed_dict={x:batch_x_test,y:batch_y_test})
        test_writer.add_summary(summary, iter)

        if iter %5000==0:
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Train Loss ",los)
        if iter %10==0:
            batch_x_test,batch_y_test = getTestBatch(iter/10)
            summary,los,pred=sess.run([merged,loss,prediction],feed_dict={x:batch_x_test,y:batch_y_test})
            test_writer.add_summary(summary, iter)
            if iter %5000==0:
                #print(batch_x_test[:5])
                print(batch_y_test[:5])
                print(pred[:5])
                print("Test Loss ",los)
                print("__________________")

        iter=iter+1

    w1,w2=sess.run([weights_1,weights_3],feed_dict={})
    np.save('weights_1', w1)
    np.save('weights_3', w2)


