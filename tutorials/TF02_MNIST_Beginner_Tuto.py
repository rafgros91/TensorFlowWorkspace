
# coding: utf-8

# # MNIST For ML Beginners

# ### MNIST Data
# The MNIST data is hosted on Yann LeCun's website. Start here with these two lines of code which will download and read in the data automatically:

# In[2]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# ##### General:
# The MNIST data is split into three parts: 
# - 55,000 data points of training data (mnist.train)
# - 10,000 points of test data (mnist.test)
# - 5,000 points of validation data (mnist.validation)
# 
# MNIST data point has two parts: an image of a handwritten digit and a corresponding label. We'll call the images "x" and the labels "y".
# 
# The training images are mnist.train.images and the training labels are mnist.train.labels.

# ##### Images:
# Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers.
# 
# We can flatten this array into **a vector of 28x28 = 784 numbers.**
# 
# The result is that **mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784].** The first dimension is an index into the list of images and the second dimension is the index for each pixel in each image. Each entry in the tensor is a pixel intensity between 0 and 1, for a particular pixel in a particular image.

# ##### Labels:
# Each image in MNIST has a corresponding label, a number between 0 and 9 representing the digit drawn in the image.
# 
# For the purposes of this tutorial, we're going to want our labels as "one-hot vectors". A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension.
# 
# Consequently, **mnist.train.labels is a [55000, 10] array (of floats??).**

# ### Softmax Regressions Bases
# If you want to assign probabilities to an object being one of several different things, softmax is the thing to do, because softmax gives us a list of values between 0 and 1 that add up to 1. Even later on, when we train more sophisticated models, the final step will be a layer of softmax.

# **evidence for a class i = Somme pour chaque index j (Wi,j * xj + bi)**
# - Wi: weights
# - bi: bias
# - j: index for summing over the pixels in our input image x
# 
# We then convert the evidence tallies into our predicted probabilities y using the "softmax" function:

# **y = softmax(evidence)**
# with   softmax(x) = normalize(exp(x))

# If we expand the equation:

# **softmax(x)i = exp(xi)/Sumj(exp(xj))**

# The exponentiation means that one more unit of evidence increases the weight given to any hypothesis multiplicatively. And conversely, having one less unit of evidence means that a hypothesis gets a fraction of its earlier weight. No hypothesis ever has zero or negative weight. **Softmax then normalizes these weights, so that they add up to one, forming a valid probability distribution.**

# **y = softmax(Wx+b)** for example y:(3x1), W:(3x3), x:(3x1), b:(3x1)

# ### Implementing the Regression

# To use TensorFlow, first we need to import it.

# In[3]:

import tensorflow as tf


# We want to be able to input any number of MNIST images, each flattened into a 784-dimensional vector. We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784]. (Here None means that a dimension can be of any length.)

# In[4]:

x = tf.placeholder(tf.float32, [None, 784])


# For machine learning applications, one generally has the model parameters be Variables. We create these Variables by giving tf.Variable the initial value, it doesn't matter very much what they initially are.
# - W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence for the difference classes 
# - b has a shape of [10] so we can add it to the output

# In[5]:

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# We can now implement our model. First, we multiply x by W with the expression tf.matmul(x, W), then add b, and finally apply tf.nn.softmax:

# In[6]:

y = tf.nn.softmax(tf.matmul(x, W) + b)


# ### Training
# One very common, very nice function to determine the loss of a model is called "**cross-entropy**." Cross-entropy arises from thinking about information compressing codes in information theory but it winds up being an important idea in lots of areas, from gambling to machine learning:
# - ** Hy'i(y) = -sum(y'ilog(yi))**   => y: predicted probability distrib, y': true distrib

# In[12]:

y_ = tf.placeholder(tf.float32, [None, 10])


# In[13]:

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# #### ??? 
# Instead, we apply tf.nn.softmax_cross_entropy_with_logits on the unnormalized logits (e.g., we call softmax_cross_entropy_with_logits on tf.matmul(x, W) + b)
# #### ???

# First, tf.log computes the logarithm of each element of y. Next, we multiply each element of y_ with the corresponding element of tf.log(y). Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over all the examples in the batch.

# Because TensorFlow knows the entire graph of your computations, it can automatically use the backpropagation algorithm to efficiently determine how your variables affect the loss you ask it to minimize.

# In[14]:

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In this case, we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5. But TensorFlow also provides many other optimization algorithms

# We can now launch the model in an InteractiveSession:

# In[15]:

sess = tf.InteractiveSession()


# We first have to create an operation to initialize the variables we created:

# In[16]:

tf.global_variables_initializer().run()


# We'll now run the training step 1000 times:

# In[17]:

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Using small batches of random data is called stochastic training -- in this case, **stochastic gradient descent.**

# ### Evaluating the Model
# First let's figure out where we predicted the correct label. tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis. For example, **tf.argmax(y,1) is the label our model thinks is most likely** for each input, while **tf.argmax(y_,1) is the correct label.** We can use **tf.equal to check if our prediction matches the truth**:

# In[18]:

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))


# That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.

# In[20]:

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Finally, we ask for our accuracy on our test data:

# In[21]:

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[ ]:



