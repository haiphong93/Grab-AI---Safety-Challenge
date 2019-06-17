#Author: Phong Nguyen
#Please check the References list for full references

import numpy as np 
import pandas as pd 
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import glob


#Data pre-processing
l =[]
for f in glob.glob('features/part-000*'):
    df = pd.read_csv(f)
    l.append(df)
X = pd.concat(l, axis=0, ignore_index=True)

y = pd.read_csv(filepath_or_buffer='labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv')
#A few of the booking IDs have multiple labels associated with them. I take the 
#first label encountered 
y = y.set_index('bookingID', drop = False)
y = y.loc[~y.index.duplicated(keep='first')]

#Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = StandardScaler().fit_transform(pd.concat([X.iloc[:,1:-2],X.Speed],axis=1).values)
scaled_features = pd.DataFrame(scaled_features,columns =  ['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y',
       'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'Speed'] )
X = pd.DataFrame(pd.concat([X.bookingID,pd.DataFrame(scaled_features),X.second],axis=1),
                           index=X.index, columns=['bookingID', 'Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y',
       'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'Speed','second'])

#Padding the data with zeros so that the sample size for each ID is 1500
#1500 was chosen because it was approximately the 80th percentile in sample length
#Code inspired by Aishwarya Singh, link: https://www.analyticsvidhya.com/blog/2019/01/introduction-time-series-classification/
X.set_index(['bookingID', 'second'], inplace=True)
X_final = []
for i in y['bookingID']:
    X_final.append(np.array(X.xs(i).sort_index(level=1)))
from keras.preprocessing import sequence
to_pad = 7561
seq_len = 1500
for i in range(0,len(X_final)):
        new_seq = []
        len_trip = len(X_final[i])
        zero_row= [0]*X_final[0].shape[1]
        n = to_pad - len_trip
        to_concat = np.repeat(zero_row, n).reshape(9, n).transpose()
        new_seq.append(np.concatenate([X_final[i], to_concat]))
        X_final[i] = np.squeeze(sequence.pad_sequences(new_seq, maxlen=seq_len,
               padding='post', dtype='float', truncating='post'))
X_final = np.array(X_final)

    
#Delete the 'Accuracy' column. It should not affect the danger
#label. It could have been deleted earlier
X_final = np.delete(X_final,0,2) 

#Train test split: 80-20, stratified
X_train, X_test, y_train, y_test = train_test_split(X_final, 
                                                    np.array(y['label']),
                                                    stratify=np.array(y['label']), 
                                                    test_size=0.2)
#Lagging. Specific lagging function from Jason Brownlee at this url: 
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
def lagging(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1] 
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	return agg
dummy_X_train = []
dummy_X_test = []

for i in range (0, len(X_train)):
    dummy_X_train.append(np.array(lagging(X_train[i],5))) # 5 steps of lagging
dummy_X_train = np.array(dummy_X_train)
X_train = np.nan_to_num(dummy_X_train)

for i in range (0, len(X_test)):
    dummy_X_test.append(np.array(lagging(X_test[i],5)))
dummy_X_test = np.array(dummy_X_test)
X_test = np.nan_to_num(dummy_X_test)

#Delete used variables to free up memory
del dummy_X_train
del dummy_X_test
del X_final

'''
#Over-sampling
Because the dataset has roughly 3 times the negative class to the positive
class, the positive instances are copied twice to the training set in order to
create a balanced dataset
'''

X_1 = []
y_1 = []
for i in range (0,X_train.shape[0]):
    if y_train[i] == 1:
        X_1.append(X_train[i])
        y_1.append(y_train[i])
X_1 = np.array(X_1)
y_1 = np.array(y_1)

X_train = np.concatenate((X_train,X_1,X_1))
y_train = np.concatenate((y_train,y_1,y_1))

#Create model
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers import Dropout, Conv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l1


#Model - CNN
#Define custom function for ROC AUC score
def roc_auc(y_true, y_pred): 
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
#Model parameters
seq_len = 1500
learning_rate = 0.0001
num_epochs = 50
batch_size = 32
adam = Adam(lr=learning_rate) 
#Build Model
#Save best model weights
chk= ModelCheckpoint('best_model.hdf5', save_best_only=True,
                     monitor='val_roc_auc', mode='max') #Save best model weights
#Callbacks
callbacks = EarlyStopping(monitor='val_roc_auc', patience=1000,
                             verbose=1, mode='max')
#Initialize model
model = Sequential()
model.add(Conv1D(filters = 64, kernel_size= 3, strides = 2, 
                  input_shape=(seq_len, 48),
                  activity_regularizer=l1(0.001)))
model.add(Dropout(0.2))
model.add(Conv1D(filters = 64, kernel_size= 3, strides = 2,
                  activity_regularizer=l1(0.001)))
model.add(Dropout(0.2))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=adam, metrics = [roc_auc])
model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size,
            validation_data=(X_test,y_test), 
            callbacks = [callbacks, chk])

#Load best weights and predict:
model.load_weights('best_model.hdf5')
model_ouput = model.predict(X_test)
'''
#Convert neural network output to 0 and 1 label.
I noticed that having the threshold for the conversion at 0.45 gives better 
score than having at the traditional 0.5 across a few models I have trained.
However, I will still leave the threshold at 0.5 for now. 
'''
label = (model_ouput>0.5).astype(int)
score = roc_auc_score(y_test, label)
print(score)

