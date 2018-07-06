
#Toy example of basic Four-node Decentralized Neural Network. Currently in progress as of 7/5/18 (does not work YET)
#by Patrick

'''TBD: 
1. Gradient limiting factor Theta_U,Theta_D (for upload/download); 
2. Gradient clipping; 
3. (Differential privacy) Laplace Noise maybe?
'''

#Helper function to swap identities of gradients from one to the other. 
#I.e. if g_W1[2] belongs to node 2 but we want to copy this gradient over to node 3 using apply_gradient, 
#we need to swapIdentity(g_W1[2], W1[3]) -> gradient for node 2, but once we send to apply_gradient,
#it sees it as gradient for node 3.
def swapIdentity(gradient, newIdentity):
    newGrad = gradient
    newGrad[1] = newIdentity
    return newGrad

#-----------main code----------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #download MNIST data (train, validate, test)

tf.reset_default_graph()

#Hyperparameters for future use:
theta_D = 1 #download rate
theta_U = 0.01 #Upload rate
#epsilon?

sz = 4 #number of Computational nodes

# Variables for our  4- way network
W1 = [tf.Variable(tf.truncated_normal([784, 500], stddev=0.1)) for i in range(sz)]
b1 = [tf.Variable(tf.zeros([500]))                             for i in range(sz)]
W2 = [tf.Variable(tf.truncated_normal([500, 100], stddev=0.1)) for i in range(sz)]
b2 = [tf.Variable(tf.zeros([100]))                             for i in range(sz)]
W3 = [tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))  for i in range(sz)]
b3 = [tf.Variable(tf.zeros([10]))                              for i in range(sz)]


#Temporary gradient banks for privacy (noise may be added in future)
g_W1 = g_W2 = g_W3 = [None for i in range(sz)]
g_b1 = g_b2 = g_b3 = [None for i in range(sz)]

grad_W1 = grad_W2 = grad_W3 = grad_b1 = grad_b2 = grad_b3 = [[None for i in range(sz)] for i in range(sz)]

#hidden layers, scores, etc.
h1 = h2= [None for i in range(sz)]
scores = loss= [None for i in range(sz)]

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

correct_predictions = [None for i in range(sz)]
accuracy = [None for i in range(sz)]

# Accuracy indicator for 4 channels
for i in range(sz):
    correct_predictions[i] = tf.equal(tf.argmax(scores[i], axis=1), tf.argmax(y[i], axis=1))
    accuracy[i] = tf.reduce_mean(tf.cast(correct_predictions[i], tf.float32))

# Initialization of all decleared Tensors (above) in Tensorflow
init = tf.global_variables_initializer()

# Training
sess = tf.Session()
sess.run(init) #initalize all variables

train_acc=[0 for i in range(sz)]
train_loss=[0 for i in range(sz)]



#for our case, WLOG (Without Loss of Generality) let's assume there are FOUR DECENTRALIZED NODES
#WLOG let's call current node A, and neighbor computational nodes B,C,and D.


#Split data into 4-ths. WLOG 4 nodes: (750,750,750,750) epochs for A,B,C, and D
for cnode in range(sz):
    for i in range(3000 // sz):

        if(g_W1[nei] != None and g_W2[nei] != None and g_W3[nei] == None):   #*****!!!Skip gradient download if no data in them initially!!!**** 
            #****repurpose gradients of B, C, D to go to A***
            for nei in range(sz):
                if(nei == cnode): #i.e. if it already is A, ignore
                    continue


                #Swap identities of neighbors' gradients with current node
                #assert(g_W1[nei][1] == W1[nei])

                grad_W1[nei][cnode] = swapIdentity(g_W1[nei], W1[cnode])
                grad_W2[nei][cnode] = swapIdentity(g_W2[nei], W2[cnode])
                grad_W3[nei][cnode] = swapIdentity(g_W3[nei], W3[cnode])
                grad_b1[nei][cnode] = swapIdentity(g_b1[nei], b1[cnode])
                grad_b2[nei][cnode] = swapIdentity(g_b2[nei], b2[cnode])
                grad_b3[nei][cnode] = swapIdentity(g_b3[nei], b3[cnode])

            #***Download gradients from nodes B,C,and D and use them to affect A's gradient.***
            for nei in range(sz):
                if(nei == cnode):
                    continue      
                opt.apply_gradients([grad_W1[nei][cnode], grad_W2[nei][cnode], grad_W3[nei][cnode], grad_b1[nei][cnode], grad_b2[nei][cnode], grad_b3[nei][cnode]])
        
        #***Take minibatch of 50, compute gradients w.r.t weights***
        x_batch, y_batch = mnist.train.next_batch(50)
        train_loss[cnode], train_acc[cnode] = sess.run([loss[cnode], accuracy[cnode]], feed_dict={X[cnode]: x_batch, y[cnode]: y_batch})
        if(i % 100 == 0):
            print("accuracy for Node: ", cnode,  train_acc[cnode])

        #upload trained weights to neighbors B, C
        g_W1[cnode], g_W2[cnode], g_W3[cnode], g_b1[cnode], g_b2[cnode], g_b3[cnode] = compute_gradients(loss=loss[cnode], var_list=[W1[cnode], W2[cnode], W3[cnode], b1[cnode], b2[cnode], b3[cnode]])

        #set some weights to Zero (for privacy); keeping largest theta_U% of weights
        #TBD LATER


#Print out your accuracy on the test set at the end.
print("Final accuracy on test set:")
print(sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels}))
