
# coding: utf-8

# # Getting Started with TensorFlow

# ### Importing the module

# In[1]:

#import statement
import tensorflow as tf
import numpy as np


# ### Computational Graph - Definition
# You might think of TensorFlow Core programs as consisting of two discrete sections:
# - Building the computational graph.
# - Running the computational graph.
# 
# A computational graph is a series of TensorFlow operations arranged into a graph of nodes. Let's build a simple computational graph. **Each node takes zero or more tensors as inputs and produces a tensor as an output.** 

# ### TensorFlow Constants
# One type of node is a constant. Like all **TensorFlow constants, it takes no inputs, and it outputs a value it stores internally.** We can create two floating point Tensors node1 and node2 as follows:

# In[2]:

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

# ### Run the computational graph within a session
# To actually evaluate the nodes, we must run the computational graph within a session. **A session encapsulates the control and state of the TensorFlow runtime.**
# 
# The following code creates a Session object and then invokes its run method to run enough of the computational graph to evaluate node1 and node2. By running the computational graph in a session as follows:

# In[3]:

sess = tf.Session()
print(sess.run([node1, node2]))


# ### Combining Tensor nodes with operations 

# For example, we can add our two constant nodes and produce a new graph as follows:

# In[4]:

#Add Operation
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))


# ### Graph with placeholders
# A graph can be parameterized to accept external inputs, known as placeholders. A placeholder is a promise to provide a value later.

# In[5]:

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)


# We can evaluate this graph with multiple inputs by using the feed_dict argument to the run method to feed concrete values to the placeholders:

# In[6]:

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))


# In[7]:

#Example with * operation:
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))


# ### Use of Variables
# To make the model trainable, we need to be able to modify the graph to get new outputs with the same input. **Variables allow us to add trainable parameters to a graph.** They are **constructed with a type and initial value:**

# In[8]:

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b


# To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:

# In[9]:

init = tf.global_variables_initializer()
sess.run(init)


# It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables. **Until we call sess.run, the variables are uninitialized.**
# 
# Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously as follows:

# In[10]:

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))


# ### Evaluation of the model
# To evaluate the model on training data, we need a **y** placeholder to provide the **desired values**, and we need to write a loss function.
# 
# - A **loss function measures how far apart the current model is from the provided data.** We'll use a standard loss model for **linear regression**, which **sums the squares of the deltas between the current model and the provided data.**
# *linear_model - y* creates a vector where each element is the corresponding example's error delta. We call tf.square to square that error. 
# - Then, we **sum all the squared errors** to create a single scalar that abstracts the error of all examples **using tf.reduce_sum**:

# In[11]:

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


#  We can change W and b accordingly (but the goal is to do this automatically):

# In[12]:

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


# ### Train:
# TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function. The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative of loss with respect to that variable.

# In[13]:

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


# In[14]:

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))


# ### Complete Program:

# In[15]:

import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


# ### Use of tf.estimator
# tf.estimator is a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following:
# 
# - running training loops
# - running evaluation loops
# - managing data sets

# In[16]:

import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one numeric feature. There are many
# other types of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)


# ### Example of use of tf.estimator with custom model
# tf.estimator does not lock you into its predefined models. Suppose we wanted to create a custom model that is not built into TensorFlow. We can still retain the high level abstraction of data set, feeding, training, etc. of tf.estimator. For illustration, we will show how to implement our own equivalent model to LinearRegressor using our knowledge of the lower level TensorFlow API.
# 
# To define a custom model that works with tf.estimator, we need to use tf.estimator.Estimator. tf.estimator.LinearRegressor is actually a sub-class of tf.estimator.Estimator. Instead of sub-classing Estimator, we simply provide Estimator a function model_fn that tells tf.estimator how it can evaluate predictions, training steps, and loss. The code is as follows:

# In[17]:

import numpy as np
import tensorflow as tf

# Declare list of features, we only have one real-valued feature
def model_fn(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # EstimatorSpec connects subgraphs we built to the
  # appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# train
estimator.train(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)


# In[ ]:



