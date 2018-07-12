#TODO:add more error bars (for all of graphs) 
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
%matplotlib inline

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

#Helper function to swap identities of gradients from one to the other.
def swapIdentity(gradient, newIdentity):
    return (gradient[0], newIdentity) #return new tuple with the identity replaced

def shp(x):
    return np.shape(x) 

#-----------------------Initialization---------------------------
#Hyperparameters:
theta_D = 1 #download rate
theta_U = 0.01 #Upload rate

#sz = 4 # size of network
node_sz = []
accs = []

node_sz_fully = []
accs_fully = []

node_sz_indiv = []
accs_indiv = []
plots = 0

for architecture in range(3): #0 = dense graph N(N-1)/2 connections, #1 = sparse graph (N connections), #3 = independent 
    for sz in range(4,16):
        
        tf.reset_default_graph()

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
        opt =[None for i in range(sz)]

        for i in range(sz):
            opt[i] = tf.train.GradientDescentOptimizer(0.05)
            train_one_step[i] = opt[i].minimize(loss[i])



        correct_predictions = [0.0 for i in range(sz)]
        accuracy = [0.0 for i in range(sz)]

        # Accuracy indicator for 4 channels
        for i in range(sz):
            correct_predictions[i] = tf.equal(tf.argmax(scores[i], axis=1), tf.argmax(y[i], axis=1))
            accuracy[i] = tf.reduce_mean(tf.cast(correct_predictions[i], tf.float32))

        # Initialization of all decleared Tensors (above) in Tensorflow
        init = tf.global_variables_initializer()

        # Training
        sess = tf.Session()
        sess.run(init) #initalize all variables

        #Gradient "bank"

        train_acc=[0.0 for i in range(sz)]
        train_loss=[0.0 for i in range(sz)]
        test_acc = [0.0 for i in range(sz)]

        send_gradients = [None for i in range(sz)]

        #-----------------Build the gradients transfer graph---------------------
        for cnode in range(sz):
            #compute local gradients
            g_W1[cnode], g_W2[cnode], g_W3[cnode], g_b1[cnode], g_b2[cnode], g_b3[cnode] = opt[cnode].compute_gradients(
                    loss=loss[cnode], var_list=[W1[cnode], W2[cnode], W3[cnode], b1[cnode], b2[cnode], b3[cnode]])

            #EXperimennt around with either 1)dense, N(N-1)/2 connections, or 2)sparse graph, N connections
            #for nei in range(sz): #Dense graph
            if(architecture == 1):
                neighbors = range(sz)
            elif(architecture == 2):
                neighbors = [(cnode+1) % sz, cnode, (sz + cnode - 1) % sz]
            else:
                neighbors = [cnode]
            
            allgrads = []
            for nei in neighbors:
                #Distribute local gradients from current node to neighboring nodes
                grad_W1[cnode][nei] = swapIdentity(g_W1[cnode], W1[nei])
                grad_W2[cnode][nei] = swapIdentity(g_W2[cnode], W2[nei])
                grad_W3[cnode][nei] = swapIdentity(g_W3[cnode], W3[nei])
                grad_b1[cnode][nei] = swapIdentity(g_b1[cnode], b1[nei])
                grad_b2[cnode][nei] = swapIdentity(g_b2[cnode], b2[nei])
                grad_b3[cnode][nei] = swapIdentity(g_b3[cnode], b3[nei])
                allgrads += [grad_W1[cnode][nei],grad_W2[cnode][nei],grad_W3[cnode][nei],
                             grad_b1[cnode][nei],grad_b2[cnode][nei],grad_b3[cnode][nei]]

            send_gradients[cnode] = opt[cnode].apply_gradients(allgrads)
        #------------------------------Begin Training------------------------------

        #NOTE only for the first node (cnode = 0)
        x_train = []
        y_train = []

        x_test = []
        y_test = []

        #Split data into 4-ths. (50,50,50,50)

        for i in range(2000 // sz):

            for cnode in range(sz):      
                #---------download minibatch of 50 & get gradients-----------------------
                X_batch, y_batch = mnist.train.next_batch(50)

                #send gradients computed from current node to neighbors
                if(architecture == 1):
                    neighbors = range(sz)
                elif(architecture == 2):
                    neighbors = [(cnode+1) % sz, cnode, (sz + cnode - 1) % sz]
                else: #arch = 0: indiviaul
                    neighbors = [cnode]
                    
                for nei in neighbors:
                #for nei in range(sz):
                    sess.run(send_gradients[cnode], feed_dict={X[cnode]: X_batch, y[cnode]: y_batch})



                if(i % 50 == 0):
                    train_acc[cnode] = sess.run(accuracy[cnode], feed_dict={X[cnode]: X_batch, y[cnode]: y_batch})
                    test_acc[cnode] = sess.run(accuracy[cnode], feed_dict={X[cnode]: mnist.test.images, y[cnode]: mnist.test.labels})
                    #PLOT ACCURACY OF FIRST NODE
                    if(cnode == 0):
                        x_train += [i]
                        y_train += [train_acc[cnode]]

                        x_test += [i]
                        y_test += [test_acc[cnode]]
                    print("accuracy for Node: ", cnode,  "  is: ",train_acc[cnode],  "  at epoch: ", i, " test acc: ", test_acc[cnode])

        #after done with training (for i loop), get FINAL test set accuracy for export.
        plots+=1
        plt.figure(plots)
        
        arch = ''
        if(architecture == 1):
            arch = "Dense network"
        elif(architecture == 2):
            arch = "Sparse network"
        else:
            arch = "Independent networks"
        
        plt.title("Number of Nodes: " +str(sz) + " , " + arch)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(x_train, y_train, 'r', x_test, y_test, 'b')

        maxacc = 0
        # Make sure to print out your accuracy on the test set at the end.
        print("Final accuracy on test set:")
        for i in range(sz):
            accy =  sess.run(accuracy[i], feed_dict={X[i]: mnist.test.images, y[i]: mnist.test.labels})
            print(i, " has accuracy: ",accy)
            maxacc = max(maxacc, accy)
        
        
        if(architecture == 1): #arch = 1: fully
            node_sz_fully += [sz]
            accs_fully += [maxacc]
        elif(architecture == 2): #arch = 2: sparse graph
            node_sz += [sz]
            accs += [maxacc]
        else: #arch = 0: individual
            node_sz_indiv += [sz]
            accs_indiv += [maxacc]

#FINAL PLOT for the #of Agents (Node size) vs. accuracy (max of all agents) 
plt.figure(plots+1)
plt.xlabel('Node size')
plt.ylabel('Accuracy')
#GREEN: sparse results RED: dense results BLUE: individual results
plt.plot(node_sz, accs, 'g', node_sz_fully, accs_fully, 'r', node_sz_indiv, accs_indiv, 'b')
