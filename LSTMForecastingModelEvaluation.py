#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
from pandas import DataFrame, read_csv, concat
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from matplotlib import pyplot as plt
from numpy import concatenate, reshape
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler


# In[2]:


# Series to Supervised Learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
    # print("I: ",i)
        cols.append(df.shift(i))
        # print("Column: ",cols)
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # print("Names: ",names)
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        # print("COls: ",cols)
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # print("Names: ",names)

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg


# In[3]:


# Read Data and Extract Values
dataset = read_csv('cleanedBTCValues - Multiplied.csv') 
nrows = dataset.shape[0]
values = dataset.iloc[:,3:].values #Getting values - Total Sentiment and BTC Values
valuesSentiment = dataset.iloc[:,3:].values #Getting total sentiment scores only
valuesBTC = dataset.iloc[:,4:].values #Getting vwap scores only


# In[4]:


# Scaling
scaler = MinMaxScaler(feature_range = (0,1))
scaler = scaler.fit(values)
scaled = scaler.fit_transform(values)


# In[5]:


# Convert Series to Supervised Data
reframed = series_to_supervised(scaled, 1, 1)

# Drop previous sentiment
reframed=reframed.drop(columns=['var1(t-1)'])

#Splitting data into train and test sets
reframedValues = reframed.values


n_train_days = int(0.9*nrows) #90% data is train, 10% test
train = reframedValues[:n_train_days, :]
test = reframedValues[n_train_days+1:nrows, :]

#Assigning inputs and output datasets
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

#Reshaping input to be 3 dimensions (samples, timesteps, features)
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


# In[6]:


#Building LSTM Neural Network model
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]))) #Recurrent Layer

model.add(Dropout(0.4)) #Dropout Layer
model.add(Dense(15, activation = 'tanh')) #Fully Connected Layer
model.add(Dense(1, activation = 'sigmoid')) #Output Layer
model.compile(loss='mae', optimizer= 'adam', metrics=['acc']) #Compiling the model

# Uncoomen below line to get summary of the model
# print(model.summary(line_length=None, positions=None, print_fn=None))


#Fitting model
history = model.fit(train_X, train_y, epochs = 200, batch_size=25, validation_data=(test_X, test_y), verbose=2, shuffle=False) #Best so far: 100 neurons, epochs = 400, batch_size = 53
print(history.history)


# In[7]:


# Predicition
model_prediction = model.predict(test_X)


# In[8]:


# Reshae Test_X
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


# In[9]:


# BTC Value Scaling
scalerBTC = MinMaxScaler(feature_range = (0,1))
scalerBTC = scaler.fit(values)
scaledBTC = scaler.fit_transform(values)


# In[10]:


# Inverse Scale
scaler = MinMaxScaler(feature_range = (0,1))
scaler = scaler.fit(valuesBTC)
model_prediction_unscale = scaler.inverse_transform(model_prediction)

predictedValues = reshape(model_prediction_unscale, model_prediction_unscale.shape[0])
actualValues = valuesBTC[n_train_days+1:]
actualValues = reshape(actualValues, actualValues.shape[0])


# In[11]:


#Plotting training loss vs validation loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()


# In[12]:


#Visualising Results (Actual vs Predicted)
plt.plot(actualValues, color = 'red', label = 'Actual Bitcoin VWAP')
plt.plot(predictedValues, color = 'blue', label = 'Predicted Bitcoin VWAP') #[1:38]
plt.title('Bitcoin VWAP Prediction')
plt.xlabel('Time Interval (1 interval = 3.5 hours)')
plt.ylabel('VWAP')
plt.legend()

# Uncomment below line to save the figure
# plt.savefig("Check Results.png", dpi=700)

plt.show()


# In[13]:


actual= DataFrame(actualValues, columns= ['Actual Value'])
predicted=DataFrame(predictedValues, columns= ['Predicted Value'])


# In[14]:


#Calculating RMSE and MAE
errorDF=concat([actual,predicted], axis=1)
errorDF.dropna(inplace=True)
rmse = sqrt(mean_squared_error(errorDF.iloc[:,0], errorDF.iloc[:,1]))
mae = mean_absolute_error(errorDF.iloc[:,0], errorDF.iloc[:,1])
print('Test MAE: %.3f' % mae)
print('Test RMSE: %.3f' % rmse)


# In[15]:


# Write to csv
timestamp = DataFrame(dataset['timestamp'][n_train_days:], columns= ['timestamp'])
timestamp.reset_index(drop=True, inplace=True)
results=concat([timestamp,actual,predicted], axis=1)
results.dropna(inplace=True)
results.to_csv("Results.csv", index= False)


# In[ ]:




