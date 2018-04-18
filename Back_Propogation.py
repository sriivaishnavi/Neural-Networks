
# coding: utf-8

# In[43]:


#--- back propogation ----Sai Sree Vaishnavi---012415130
import numpy as np
import math


# In[44]:


X = np.array([[1,1,1],[1,0,1],[0,1,1],[0,0,1]])
Y = np.array([[0],[1],[1],[0]])


# In[45]:


#randomly choosing weights and bias
w1 = np.random.randn(3,4)
w3 = np.random.randn(4,1)
print(w1)
print(w3)
b1 = np.random.randn(4,4)
b3 = np.random.randn(4,1)
print(b1)
print(b3)


# In[46]:



def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-1.0*x))
    return s


# In[47]:


#hidden layer
h1 = np.dot(X,w1) + b1
h1= sigmoid(h1)
print(h1)


# In[48]:


output = np.dot(h1,w3) + b3
print(output)


# In[49]:


output


# In[50]:


def derivative(output):
    return output * (1.0 - output)
for i in range(0,100):
    dout=sigmoid(output)*(1-sigmoid(output))
    #dw=sigmoid(w1)*(1-sigmoid(w1))
    
    w1=w1-dw

    h1 = sigmoid(np.dot(X,w1) + b1)
    output = np.dot(h1,w3) + b3
output


# In[51]:


error = (Y - output) * derivative(output)


# In[52]:


# Define the number of epochs for learning
epochs = 2000

# Initialize the weights with random numbers
w01 = np.random.random((len(X[0]), 5))
w12 = np.random.random((5, 1))

# Start feeding forward and backpropagate *epochs* times.
for epoch in range(epochs):
    # Feed forward
    z_h = np.dot(X, w1)
    a_h = sigmoid(z_h)

    z_o = np.dot(a_h, w3)
    a_o = sigmoid(z_o)

    # Calculate the error
    a_o_error = ((1 / 2) * (np.power((a_o - Y), 2)))

    # Backpropagation
    ## Output layer
    delta_a_o_error = a_o - Y
    delta_z_o = sigmoid(a_o)
    delta_w3 = a_h
    delta_output_layer = np.dot(delta_w3.T,(delta_a_o_error * delta_z_o))

    ## Hidden layer
    delta_a_h = np.dot(delta_a_o_error * delta_z_o, w3.T)
    delta_z_h = sigmoid(a_h)
    delta_w1 = X
    delta_hidden_layer = np.dot(delta_w1.T, delta_a_h * delta_z_h)

    w1 = w1 - delta_hidden_layer
    w3 = w3 - delta_output_layer


# In[53]:


print(a_o)

