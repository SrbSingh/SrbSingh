#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


df=pd.read_csv("C:\\Users\\Lenovo\\Desktop\\homeprices.csv")
df


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df['area'],df['price'],color='red',marker='+')


# In[4]:


reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price) #training machine


# In[5]:


reg.predict([[3300]])


# In[6]:


reg.coef_


# In[7]:


reg.intercept_


# In[8]:


135.78767123*3300+180616.43835616432


# In[9]:


reg.predict([[5000]])


# In[10]:


135.78767123*5000+180616.43835616432


# In[11]:


d=pd.read_csv("C:\\Users\\Lenovo\\Desktop\\areas.csv")
d


# In[13]:


d.head(13)


# In[15]:


p=reg.predict(d)


# In[16]:


d['prices']=p


# In[17]:


d


# In[62]:


d.to_csv("C:\\Users\\Lenovo\\Desktop\\prediction.csv")


# In[63]:


d.to_csv("C:\\Users\\Lenovo\\Desktop\\prediction.csv",index=False)


# In[ ]:




