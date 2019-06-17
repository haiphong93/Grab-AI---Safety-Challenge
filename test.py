"""
Use this to test model with pre-loaded weights
"""
import numpy as np 
import pandas as pd 
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import glob

#Data pre-processing
l =[]
for f in glob.glob('test_features/part-000*'):
    df = pd.read_csv(f)
    l.append(df)
X_test = pd.concat(l, axis=0, ignore_index=True)

y_test = pd.read_csv(filepath_or_buffer='test_labels/part-000*')
#Same with training label, keep the first encountered label
y_test = y_test.set_index('bookingID', drop = False)
y_test = y_test.loc[~y_test.index.duplicated(keep='first')]

#Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = StandardScaler().fit_transform(pd.concat([X_test.iloc[:,1:-2],
                                X_test.Speed],axis=1).values)
scaled_features = pd.DataFrame(scaled_features,
                               columns =  ['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y',
                                           'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'Speed'] )
X_test = pd.DataFrame(pd.concat([X_test.bookingID,pd.DataFrame(scaled_features),X_test.second],axis=1),
                           index=X_test.index, columns=['bookingID', 'Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y',
       'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'Speed','second'])

X_test.set_index(['bookingID', 'second'], inplace=True)
X_test_final = []
for i in y_test['bookingID']:
    X_test_final.append(np.array(X_test.xs(i).sort_index(level=1)))
from keras.preprocessing import sequence
to_pad = 7561
seq_len = 1500
for i in range(0,len(X_test_final)):
        new_seq = []
        len_trip = len(X_test_final[i])
        zero_row= [0]*X_test_final[0].shape[1]
        n = to_pad - len_trip
        to_concat = np.repeat(zero_row, n).reshape(9, n).transpose()
        new_seq.append(np.concatenate([X_test_final[i], to_concat]))
        X_test_final[i] = np.squeeze(sequence.pad_sequences(new_seq, maxlen=seq_len,
               padding='post', dtype='float', truncating='post'))

X_test_final = np.delete(X_test_final,0,2)
#Lagging
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

dummy_X_test = []

for i in range (0, len(X_test)):
    dummy_X_test.append(np.array(lagging(X_test[i],5)))
dummy_X_test = np.array(dummy_X_test)
X_test = np.nan_to_num(dummy_X_test)
del dummy_X_test
del X_test_final

#Create model
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers import Dropout, Conv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.optimizers import Adam
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
#Initialize model
model = Sequential()
model.add(Conv1D(filters = 64, kernel_size= 3, strides = 2, 
                  input_shape=(seq_len, 48),
                  activity_regularizer=l1(0.001)))
model.add(Dropout(0.1))
model.add(Conv1D(filters = 64, kernel_size= 3, strides = 2,
                  activity_regularizer=l1(0.001)))
model.add(Dropout(0.1))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=adam, metrics = [roc_auc])

#Load best weights and predict:
model.load_weights('best_model.hdf5')
label = model.predict(X_test)
score = roc_auc_score(y_test, (label>0.5).astype(int))
print(score)