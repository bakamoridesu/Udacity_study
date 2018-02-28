import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import pandas as pd
from sklearn.utils import shuffle

# TODO: Load traffic signs data.
data_file = 'train.p'
nb_classes = 43
#rate = 0.001

with open(data_file, mode='rb') as f:
    data = pickle.load(f)
X_data, y_data = data['features'], data['labels']

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))

resized = tf.image.resize_images(x, (227, 227)) 
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
weights = tf.Variable(tf.random_normal(shape))
biases = tf.Variable(tf.zeros(nb_classes))

logits = tf.add(tf.matmul(fc7, weights), biases)

# TODO: Define loss, training, accuracy operations.

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[weights, biases])

# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (accuracy * len(batch_x))
    return total_loss/num_examples , total_accuracy/num_examples

# TODO: Train and evaluate the feature extraction model.

EPOCHS = 5
BATCH_SIZE = 128
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        #training_accuracy = evaluate(X_train, y_train, sess)    
        validation_loss, validation_accuracy = evaluate(X_valid, y_valid, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", validation_loss)
        print("Validation Accuracy =", validation_accuracy)
        print("")
           
    print("Training finished")