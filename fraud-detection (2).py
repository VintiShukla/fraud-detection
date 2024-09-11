#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:


data=pd.read_csv("Fraud.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.isna().sum()


# In[7]:


#0 in each column means that there are no null values


# In[8]:


data.isFraud.value_counts()


# In[9]:


data.isFlaggedFraud.value_counts()


# In[10]:


#this means 1142 are true in isFraud column and none are true in isFlaggedFraud column


# In[11]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data['type']=label_encoder.fit_transform(data["type"])


# In[12]:


data= data.drop(["step","nameOrig", "nameDest","isFlaggedFraud"], axis=1)
print(data)
#these columns have no more relevant contribution to the model hence removing it will make the data much more easier to use


# In[13]:


x,y=data.loc[:, data.columns != "isFraud"], data["isFraud"]


# In[14]:


#Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)  #random state value means that training and test data wont shuffle randomly each time it is run. It will use the same dataset.  0.2 means 20% will be test set and 80% training.
print(x_train)


# In[15]:


print(y_train)


# In[16]:


print(y_test)


# In[17]:


print(x_test)


# In[18]:


#now we normalize the rest of the column values (amount, oldbalanceorig, newbalanceorig, oldbalancedest, new balancedest),
#because when there is drastic difference in values, machine learning model will have bias
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train_array = x_train.values
x_train_array[:, 1:] = sc.fit_transform(x_train_array[:, 1:])

x_test_array = x_test.values
x_test_array[:, 1:] = sc.transform(x_test_array[:, 1:])

x_train = pd.DataFrame(x_train_array, columns=x_train.columns)
x_test = pd.DataFrame(x_test_array, columns=x_test.columns)


# In[19]:


print(x_train)


# In[20]:


print(x_test)


# In[21]:


#Creating a logistic regression model to analyze data and create prediction model

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[22]:


#Predicting test set values
y_pred = classifier.predict(x_test)
y_test_array = y_test.values

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test_array.reshape(len(y_test_array),1)),1))
#creates a side by side comparison of predicted result and actual result of targeted output(isFraud)


# In[23]:


#creating confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[27]:


#above accuracy score is the accuracy of the model.
import matplotlib.pyplot as plt
from sklearn import metrics
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.show()


# In[28]:


#Confusion matrix shows that from the test set, there are 209481 true positives, 47 true negatives, 20 false positives and 167 false negatives.
#this means that there 187 values that were inaccurately predicted in the model, whereas the rest of the predictions stand correct


# In[ ]:




