# Imports

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import random

from pylab import rcParams
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses


sns.set(style="darkgrid", palette='muted', font_scale=1.2)
whimsy_pallette = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(whimsy_pallette))
rcParams['figure.figsize'] = 12, 8
st.set_option('deprecation.showPyplotGlobalUse', False)

# Open data
data_path = 'data/ecg.csv'
df = pd.read_csv(data_path, header=None)

raw_data = df.values

# Separate x and y
# The last element contains the labels
target = raw_data[:, -1]

# The other data points are the electrocadriogram data
features = raw_data[:, 0:-1]

# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=21,
)

# Convert to tensors
min_val = tf.reduce_min(x_train)
max_val = tf.reduce_max(x_train)

x_train = (x_train - min_val) / (max_val - min_val)
x_test = (x_test - min_val) / (max_val - min_val)

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# Set labels to boolean and split up normal and abnormal heartbeats
y_train = y_train.astype(bool)
y_test = y_test.astype(bool)

x_norm_train = x_train[y_train]
x_norm_test = x_test[y_test]

x_abnorm_train = x_train[~y_train]
x_abnorm_test = x_test[~y_test]
 
# Some Functions
def find_threshold(model, x_train_scaled):
  reconstructions = model.predict(x_train_scaled)
  # provides losses of individual instances
  reconstruction_errors = tf.keras.losses.mae(reconstructions, x_train_scaled)

  # threshold for anomaly scores
  threshold = np.mean(reconstruction_errors.numpy()) \
      + np.std(reconstruction_errors.numpy())
  return threshold

def get_predictions(model, x_test_scaled, threshold):
  predictions = model.predict(x_test_scaled)
  # provides losses of individual instances
  errors = tf.keras.losses.mae(predictions, x_test_scaled)
  # 0 = anomaly, 1 = normal
  anomaly_mask = pd.Series(errors) > threshold
  preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
  return preds

def print_stats(predictions, labels):
  print(f"Accuracy = {accuracy_score(labels, preds)}")
  print(f"Precision = {precision_score(labels, preds)}")
  print(f"Recall = {recall_score(labels, preds)}")

def plot_ecg(title, idx_num):
    
    encoded_imgs = encoder(x_test).numpy()
    decoded_imgs = decoder(encoded_imgs).numpy()
   
    fig = plt.figure(figsize=(10,4))
    fig.tight_layout()

    plt.plot(x_test[idx_num],'darkblue',)
    plt.plot(decoded_imgs[idx_num],'crimson')
    plt.fill_between(np.arange(140), decoded_imgs[idx_num], x_test[idx_num], color='#F9E79F' )
    plt.title(f'ECG #{idx_num+1}')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    
    plt.suptitle(title, 
             x=(41-(len(title)/4))/100, y=.95, 
             horizontalalignment='left', 
             verticalalignment='bottom', 
             fontweight ="bold")
    # plt.savefig(save_file_name, dpi=300, pad_inches=0.1,)
    # plt.show()
    st.pyplot()
  
# Autoencoder model with best parameters
encoder = Sequential([
      Dense(28, activation='relu'),
      Dropout(0.1),
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(60, activation='relu'),
      Dropout(0.1),
      Dense(8, activation='relu')
    ])
decoder = Sequential([
      Dense(56, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(60, activation='relu'),
      Dropout(0.1),
      Dense(140, activation='sigmoid')
    ])

best_model = keras.models.Sequential([encoder, decoder])
best_model.compile(loss='mae', optimizer=Adam(learning_rate = 0.001))

history = best_model.fit(
    x_norm_train,
    x_norm_train,
    epochs=100,
    verbose=0,
    batch_size=512,
    validation_data=(x_test, x_test)
)    


# Calculate thresholds and get predictions
threshold = find_threshold(best_model, x_norm_train)
preds = get_predictions(best_model, x_test, threshold)

#Streamlit stuff
st.title("ECG Demo")
st.write("#### Move the slider to the ECG number to see the model prediction")
ecg_num = st.slider('ECG Number', min_value=1, max_value=len(y_test), value=int(len(y_test)/2), step=1)
idx_num = ecg_num - 1


plot_ecg("Original ECG vs Reconstruction", idx_num=idx_num)


if preds[idx_num] == True:
    st.write("#### The model predicts a normal ECG")
else: 
    st.write("#### The model predicts an anomalous ECG")
st.image('images/Medic_Logo.png', width=200)
