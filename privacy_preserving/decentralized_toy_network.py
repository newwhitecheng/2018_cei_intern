import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

#Helper function to swap identities of gradients from one to the other.
def swapIdentity(gradient, newIdentity):
    return (gradient[0], newIdentity) #return new tuple with the identity replaced

def shp(x):
    return np.shape(x) 

tf.reset_default_graph()

#-----------------------Initialization---------------------------
#Hyperparameters:
theta_D = 1 #download rate
theta_U = 0.01 #Upload rate

sz = 4 # size of network

# Variables for our  4- way network
W1 = [tf.Variable(tf.truncated_normal([784, 500], stddev=0.1)) for i in range(sz)]
b1 = [tf.Variable(tf.zeros([500]))                             for i in range(sz)]
W2 = [tf.Variable(tf.truncated_normal([500, 100], stddev=0.1)) for i in range(sz)]
b2 = [tf.Variable(tf.zeros([100]))                             for i in range(sz)]
W3 = [tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))  for i in range(sz)]
b3 = [tf.Variable(tf.zeros([10]))                              for i in range(sz)]

#Temporary gradient banks for communication between nodes (noise may be added in future)
g_W1 = [None for i in range(sz)]
g_W2 = [None for i in range(sz)]
g_W3 = [None for i in range(sz)]
g_b1 = [None for i in range(sz)]
g_b2 = [None for i in range(sz)]
g_b3 = [None for i in range(sz)]


grad_W1 = [[None for i in range(sz)] for i in range(sz)]
grad_W2 = [[None for i in range(sz)] for i in range(sz)]
grad_W3 = [[None for i in range(sz)] for i in range(sz)]
grad_b1 = [[None for i in range(sz)] for i in range(sz)]
grad_b2 = [[None for i in range(sz)] for i in range(sz)]
grad_b3 = [[None for i in range(sz)] for i in range(sz)]


#hidden layers, scores, etc. *Note there is somehow a problem with saying scores = loss = [None for i in range(sz)] so I put them in seperate lines
h1 = [None for i in range(sz)]
h2 = [None for i in range(sz)]

scores = [None for i in range(sz)]
loss = [None for i in range(sz)]

#-------------------------Define Decentralized Neural Network Structure-----------------

# Placeholders
X = [tf.placeholder(tf.float32, [None, 784])  for i in range(sz)]
y = [tf.placeholder(tf.float32, [None, 10])    for i in range(sz)]


# Deep Neural nets for 4 channels w/ loss function
for i in range(sz):
    h1[i] = tf.nn.relu(tf.matmul(X[i],  W1[i]) + b1[i])
    h2[i] = tf.nn.relu(tf.matmul(h1[i], W2[i]) + b2[i])
    scores[i] = tf.matmul(h2[i], W3[i]) + b3[i]
    loss[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores[i], labels=y[i]))

# Gradient descent for 4 channels
train_one_step = [None for i in range(sz)]
for i in range(sz):
    train_one_step[i] = tf.train.GradientDescentOptimizer(0.05).minimize(loss[i])

opt = tf.train.GradientDescentOptimizer(0.05)

correct_predictions = [0.0 for i in range(sz)]
accuracy = [0.0 for i in range(sz)]

# Accuracy indicator for 4 channels
for i in range(sz):
    correct_predictions[i] = tf.equal(tf.argmax(scores[i], axis=1), tf.argmax(y[i], axis=1)) #something wrong with the axis dimension :-( !!
    accuracy[i] = tf.reduce_mean(tf.cast(correct_predictions[i], tf.float32))

# Initialization of all decleared Tensors (above) in Tensorflow
init = tf.global_variables_initializer()

# Training
sess = tf.Session()
sess.run(init) #initalize all variables

#Gradient "bank"

train_acc=[0.0 for i in range(sz)]
train_loss=[0.0 for i in range(sz)]

#------------------------------Begin Training------------------------------

#Split data into 4-ths. (750,750,750,750)

for i in range(3000 // sz):
    
    for cnode in range(sz):
 
        #----------download neighboring nodes' gradients (in this case, all (sz choose 2) or sz(sz-1)/2 nodes )------------
        for nei in range(sz):
            #Don't double train same node
            if(nei == cnode):
                continue

            #Skip during 1st run of code
            if(g_W1[nei] == None): 
                continue

            grad_W1[nei][cnode] = swapIdentity(g_W1[nei], W1[cnode])
            grad_W2[nei][cnode] = swapIdentity(g_W2[nei], W2[cnode])
            grad_W3[nei][cnode] = swapIdentity(g_W3[nei], W3[cnode])
            grad_b1[nei][cnode] = swapIdentity(g_b1[nei], b1[cnode])
            grad_b2[nei][cnode] = swapIdentity(g_b2[nei], b2[cnode])
            grad_b3[nei][cnode] = swapIdentity(g_b3[nei], b3[cnode])
            
            opt.apply_gradients([grad_W1[nei][cnode], grad_W2[nei][cnode], grad_W3[nei][cnode], 
                                 grad_b1[nei][cnode], grad_b2[nei][cnode], grad_b3[nei][cnode]])
        #--------------------------------------------------------------
        
        
        #---------download minibatch of 50 & get gradients-----------------------
        X_batch, y_batch = mnist.train.next_batch(50)
        
        if(i % 100 == 0):
            train_acc[cnode] = sess.run(accuracy[cnode], feed_dict={X[cnode]: X_batch, y[cnode]: y_batch})
            print("accuracy for Node: ", cnode,  "  is: ",train_acc[cnode],  "  at epoch: ", i)

        #upload trained weights to neighbors B, C
        g_W1[cnode], g_W2[cnode], g_W3[cnode], 
        g_b1[cnode], g_b2[cnode], g_b3[cnode] = opt.compute_gradients(loss=loss[cnode], 
                                                                      var_list=[W1[cnode], W2[cnode], W3[cnode],
                                                                                b1[cnode], b2[cnode], b3[cnode]])
        
        sess.run(train_one_step[cnode], feed_dict={X[cnode]: X_batch, y[cnode]: y_batch})
        #---------------------------------------------------------------------

# Make sure to print out your accuracy on the test set at the end.
print("Final accuracy on test set:")
print(sess.run(accuracy[0], feed_dict={X[0]: mnist.test.images, y[0]: mnist.test.labels}))
