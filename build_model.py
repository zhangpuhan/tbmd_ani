import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.activations import elu
from keras.utils import plot_model
from keras import regularizers
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.callbacks import Callback

train = pd.read_csv("train.csv", header=None)
test = pd.read_csv("test.csv", header=None)

train = train.values
np.random.shuffle(train)
test = test.values

train_x = np.array(train[:, :-3])
train_y = np.array(train[:, -3:])

test_x = np.array(test[:, :-3])
test_y = np.array(test[:, -3:])



# seed = 7
# np.random.seed(seed)

# model = Sequential()
# model.add(Dense(384, input_dim=384, activation='elu'))
# model.add(Dense(128, activation='elu'))
# model.add(Dense(64, activation='elu'))
# model.add(Dense(3))

# model.compile(loss='mean_squared_error', optimizer='adam')

# print(model.summary())
# model.fit(train_x, train_y, validation_split=0.2, epochs=100, batch_size=100, verbose=2)

model = Sequential()
model.add(Dense(512, input_dim=384, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(3, kernel_initializer='glorot_uniform', activation='linear'))
#     #model.add(Dense(num_output, kernel_initializer='normal'))
# 	# Compile model
model.compile(loss='mean_squared_error', optimizer='adam') # metrics=?
K.set_value(model.optimizer.lr, .00001)
print(K.get_value(model.optimizer.lr))

earlystop = EarlyStopping(monitor='mean_squared_error', min_delta=0.0001, patience=10,
                          verbose=1, mode='auto')

#     return model
history = model.fit(train_x, train_y, validation_split=0.2, epochs=100, batch_size=32, verbose=2)

predicted_m = model.get_weights()[0][0][0]
predicted_b = model.get_weights()[1][0]
print("\nm=%.2f b=%.2f\n" % (predicted_m, predicted_b))

# plot metrics
his_mse = history.history['mean_squared_error']
his_mae = history.history['mean_absolute_error']
his_mape = history.history['mean_absolute_percentage_error']
his_cosine = history.history['cosine_proximity']



# evaluate model
#estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=2)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, train_X, train_Y, cv=kfold)
# print("Baseline: %.8f (%.8f) MSE" % (results.mean(), results.std()))
# numpy.savetxt('cross_score.out', results)

# pipeline.fit(train_X, train_Y)

# Save the Keras model first:
# pipeline.named_steps['mlp'].model.save('keras_model.h5')
#
# # This hack allows us to save the sklearn pipeline:
# pipeline.named_steps['mlp'].model = None
#
# # Finally, save the pipeline:
# joblib.dump(pipeline, 'sklearn_pipeline.pkl')
#
# del pipeline
#
# # Load the pipeline first:
# pipeline = joblib.load('sklearn_pipeline.pkl')
#
# # Then, load the Keras model:
# pipeline.named_steps['mlp'].model = load_model('keras_model.h5')




