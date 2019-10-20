#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


heart=pd.read_csv('C:\\Users\Admin\Desktop\\heart.csv')


# In[13]:


heart.head()


# In[16]:


x=heart.iloc[:,[2,3]].values


# In[51]:


sns.countplot(x='target',data=heart,hue='sex')


# In[29]:


X=heart.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values


# In[30]:


y=heart.iloc[:,[13]].values


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


# In[33]:


from sklearn.linear_model import LogisticRegression


# In[36]:


model2=LogisticRegression()


# In[40]:



import warnings
warnings.filterwarnings('ignore')


# In[41]:


model2.fit(X_train,y_train)


# In[45]:


predy=model2.predict((X_test))


# In[46]:


predy


# In[48]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predy)*100


# # Final accuracy =96.7
