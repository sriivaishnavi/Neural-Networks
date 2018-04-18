
# coding: utf-8

# In[6]:


#----Sai Sree Vaishnavi---012415130
import numpy as np


# In[7]:


X = np.array([[1,1,1],[1,0,1],[0,1,1],[0,0,1]])
Y = np.array([[0],[1],[1],[0]])


# In[8]:


#randomly choosing weights and bias
w1 = np.random.randn(3,4)
w3 = np.random.randn(4,1)
print(w1)
print(w3)
b1 = np.random.randn(4,4)
b3 = np.random.randn(4,1)
print(b1)
print(b3)


# In[9]:


def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-1.0*x))
    return s


# In[10]:


#hidden layer
h1 = np.dot(X,w1) + b1
h1= sigmoid(h1)
print(h1)


# In[11]:


output = np.dot(h1,w3) + b3
print(output)

