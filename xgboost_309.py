#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import graphviz
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns 
rcParams['figure.figsize'] = 12, 4
from sklearn.metrics import classification_report
from sklearn import metrics
import shap


# In[2]:


df309 = pd.read_csv('real_309_5_en.csv')
df309.shape


# In[3]:


label = df309.groupby('label')
label.size()


# In[4]:


df309.head(1)


# In[5]:


X = df309.drop(columns = ['label', 'years','mind', 'support', 'law']).values
y = df309['label'].values


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 11)

# model = XGBClassifier()

model = XGBClassifier(
 objective = 'multi:softmax',
 eta =0.03,
 n_estimators=2000,
 max_depth= 70,
 min_child_weight=30,
 gamma=0.5,
 subsample=1,
 colsample_bytree=0.1,
 reg_lambda = 2,
 reg_alpha = 1,
 nthread=4,
 scale_pos_weight=2,
 )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)

print('number of correct sample: ', num_correct_samples)
print('accuracy: ', accuracy)
print('Precision:', precision_score(y_test, y_pred, average = 'weighted'))
print('Recall:', recall_score(y_test, y_pred, average = 'weighted'))
print('F1:', f1_score(y_test, y_pred, average = 'weighted'))
print('confusion matrix: ', con_matrix)
print(classification_report(y_test,y_pred))


# In[7]:


model = XGBClassifier()
model.fit(X, y)

plot_importance(model)
pyplot.show()


# In[8]:


plt.figure(figsize=(7,6))
sns.heatmap(con_matrix, annot=True)
plt.title('XGBoost \nAccuracy:{0:.2f}'.format(accuracy))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[9]:


import pickle


# In[10]:


with open('xgboost_309_5_en', 'wb') as file:
    pickle.dump(model, file, protocol = pickle.HIGHEST_PROTOCOL)


# In[ ]:




