#!/usr/bin/env python
# coding: utf-8

# In[78]:


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


# In[79]:


df =pd.read_csv('Admission_Predict_Ver.csv')
df.head()
     


# In[80]:


df.shape


# In[81]:


df.columns


# In[82]:


df.drop(columns = ['Serial No.'],inplace=True)


# In[83]:


df.columns


# In[84]:


df['University Rating'].value_counts()


# In[85]:


#df = pd.get_dummies(df,columns=['University Rating'],drop_first=True)
df.head()


# In[86]:


X = df.drop('Chance of Admit ',axis=1)
y = df['Chance of Admit ']
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
X_train


# In[87]:


y_train


# In[88]:


X_test


# In[89]:


# Standardizing the data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[90]:


X_test_scaled


# In[91]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# In[92]:


# Building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu',
                          input_shape=(X_train.shape[1],)), # Input layer

    tf.keras.layers.Dense(32, activation='relu'), # Hidden layer

    tf.keras.layers.Dense(1)  # Output layer for regression (no activation function on output as problem is regression so will us linear activation function)
])


# In[93]:


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# In[94]:


history = model.fit(X_train_scaled,y_train,epochs=100,verbose=1,validation_split=0.2)


# In[95]:


model.layers[0].get_weights()


# In[96]:


model.layers[1].get_weights()


# In[97]:


model.layers[2].get_weights()


# In[ ]:





# In[98]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# # using early stopping criteria

# In[101]:


# Load tips dataset
prediction =pd.read_csv('Admission_Predict_Ver.csv')
prediction.head()
     
# Preprocessing
# Converting categorical variables to dummy variables
prediction.drop(columns = ['Serial No.'],inplace=True)


X = df.drop('Chance of Admit ',axis=1)
y = df['Chance of Admit ']
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)


# Standardizing the data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# Building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu',
                          input_shape=(X_train.shape[1],)), # Input layer

    tf.keras.layers.Dense(32, activation='relu'), # Hidden layer

    tf.keras.layers.Dense(1)  # Output layer for regression (no activation function on output as problem is regression so will us linear activation function)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# Define the callback function
early_stopping = EarlyStopping(patience=5)

# Train the model with the callback function
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Evaluating the model (using Mean Squared Error)
loss = model.evaluate(X_test, y_test, verbose=0)
loss

# Plotting the training and testing loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()







# In[ ]:




