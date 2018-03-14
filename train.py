import os

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.model_selection import train_test_split

from constants import *
from utils import *

# Set the desired TensorFlow output level for this example
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(42)

path = "./kddcup.data/"
filename = os.path.join(path, "kddcup.data_updated")

df = pd.read_csv(filename, na_values=['NA', '?'], names=column_names)

# Shuffle
np.random.seed(42)
df = df.reindex(np.random.permutation(df.index))
df.reset_index(inplace=True, drop=True)

protocol_types = encode_text_index(df, "protocol_type")
services = encode_text_index(df, "service")
flags = encode_text_index(df, "flag")
classes = encode_text_index(df, "classes")

x, y = to_xy(df, "classes")

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(20, input_dim=x.shape[1], activation='relu'))
model.add(Dense(10))
model.add(Dense(y.shape[1], activation='softmax'))
adam = Adam(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer=adam)
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True)  # save best model
model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor, checkpointer], verbose=1, epochs=10)
model.load_weights('best_weights.hdf5')  # load weights from best model

# Measure accuracy
pred = model.predict(x_test)
pred2 = np.argmax(pred, axis=1)
y_test2 = np.argmax(y_test, axis=1)
score = metrics.accuracy_score(y_test2, pred2)
print("Final accuracy: {}".format(score))

mult_plot_roc(pred, y_test, classes)
