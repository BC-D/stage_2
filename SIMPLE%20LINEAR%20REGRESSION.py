#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use ('fivethirtyeight')
import seaborn as sns

from sklearn import datasets, linear_model

from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split


# In[17]:


pd.__version__


# In[31]:


df = pd.read_csv(r"C:/Users/HP/Desktop/energydata.csv")
print (df)
df.head()


# In[ ]:


from pandas import ExcelWriter
from pandas import ExcelFile


# In[39]:


df = pd.read_csv("C:/Users/HP/Desktop/energydata.csv", index_col=0)


# In[40]:


df


# In[35]:


df.head()


# In[37]:


df.info()


# In[38]:


df.describe()


# In[41]:





# In[1]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha= 0.001 )
lasso_reg.fit(x_train, y_train)


# In[2]:


def get_weights_df(model, feat, col_name) :


# In[3]:


weights = pd.Series(model.coef_, feat.columns).sort_values()
weights_df = pd.DataFrame(weights).reset_index()
weights_df.columns = [ 'Features' , col_name]
weights_df[col_name].round( 3 )
return weights_df


# In[4]:


import numpy as np
rss = np.sum(np.square(y_test - predicted_values))
round(rss, 3 )


# In[ ]:




