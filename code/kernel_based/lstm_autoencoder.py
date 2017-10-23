import numpy as np
from keras.layers import Input, LSTM, RepeatVector, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

# GLOBALS
OBSER_LEN = 51
LONGEST_STAY = 20
PATIENTS = 19275

def process_patient_data():
    
    print ('reading csv ...')
    data = np.genfromtxt('../../Dataset/Sepsis_imp.csv', dtype=float, delimiter=',', skip_header=1)
    
    print ('normalizing data ...')
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # remove intervention
    print ('removing intervention ...')
    interventions = np.setdiff1d(np.arange(47, 57), [51,52,54,55,56])
    data = np.delete(data, interventions, axis=1)
    #data = np.delete(data, [0,1,2], axis=1)
    
    patient = {}
    print ('building map ...')
    for i, row in enumerate(data):
        icuid = row[1]
        if icuid not in patient:
            # state_id, action, outcome
            patient[icuid] = {}
            #patient[icuid]['demographics'] = row[3:7]
            patient[icuid]['histories'] = [row[3:]]
        else:
            patient[icuid]['histories'].append(row[3:])

    return patient

def patient_data_to_train_x(patient):
    
    X = np.zeros((len(patient.keys()), LONGEST_STAY, OBSER_LEN))
    Y = np.zeros((1, len(patient.keys())))
    for i, icuid in enumerate(patient.keys()):
        if patient[icuid]['histories'][-1][-2] < 0:
            Y[:,i] = 0
        else:
            Y[:,i] = 1
        for j, hist in enumerate(patient[icuid]['histories']):
            X[i, j] = hist
    return X, Y

BATCH_SIZE = 128

def lstm_autoencoder():
    
    inputs = Input(shape=(LONGEST_STAY, OBSER_LEN))
    
    encoded = LSTM(128, input_shape=(LONGEST_STAY, OBSER_LEN))(inputs)
    #encoded = LSTM(128, input_shape=(LONGEST_STAY, OBSER_LEN), return_sequences=True)(inputs)
    #encoded = LSTM(128)(encoded)
    
    decoded = RepeatVector(LONGEST_STAY)(encoded)
    
    #decoded = LSTM(256, return_sequences=True)(decoded)
    #decoded = LSTM(128, return_sequences=True)(decoded)
    decoded = LSTM(OBSER_LEN, return_sequences=True)(decoded)
    
    sequence_autoencoder = Model(inputs, decoded)
    sequence_autoencoder.compile(optimizer='adam', loss='mse')
    
    encoder = Model(inputs, encoded)
    
    return sequence_autoencoder, encoder

def train_autoencoder(autoencoder, x):
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    early_stop = EarlyStopping(monitor='val_loss',min_delta=0, patience=2,verbose=0, mode='auto')
    autoencoder.fit(x, x,
                epochs=20,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_split=0.2, verbose=1, callbacks=[tensorboard, early_stop])


def cluster_patients(patients_rep, k=20):
    
    kmeans = MiniBatchKMeans(n_clusters=k)
    print ('clustering ...')
    kmeans.fit(patients_rep)
    print ('done clustering')
   
    return kmeans

if __name__ == '__main__':

	patient = process_patient_data()
	x, y = patient_data_to_train_x(patient)
	x_train, x_test, _, _ = train_test_split(x, x, test_size=0.1, random_state=42)

	autoencoder, encoder = lstm_autoencoder()
	train_autoencoder(autoencoder, x_train)
	# evaluate model
	autoencoder.evaluate(x_test, x_test)

	embeddings = encoder.predict_on_batch(x)

	representation_train = encoder.predict_on_batch(x_train) # 90%
	representation_test =  encoder.predict_on_batch(x_test) # 10%

	kmeans = cluster_patients(representation_train)
	labels = kmeans.labels_
	test_labels = kmeans.predict(representation_test)
