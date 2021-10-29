import keras
from keras.layers import Dense
import numpy as np
from keras.datasets import mnist

#### Load and normalize data #### 
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape(trainX.shape[0], 784).astype('float32') / 255.
testX = testX.reshape(testX.shape[0], 784).astype('float32') / 255.

#### Autoencoder 1 ####
seq_input = keras.Input(shape=(784,))

# Encoder
layers_encode = Dense(50, kernel_initializer='glorot_uniform', activation =
                      'relu')(seq_input)
layers_encode = Dense(2, kernel_initializer='glorot_uniform', activation =
                      'relu')(layers_encode)
# Decoder
layers_decode = Dense(784, kernel_initializer='glorot_uniform',
                      activation = 'relu')(layers_encode)
layers_decode = Dense(784)(layers_decode)

# Autoencoder
autoencoder = keras.Model(seq_input, layers_decode)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Settings for training
eta = 0.001
mini_batch_size = 8192
epochs = 800 # 800

# Train autoencoder
autoencoder.fit(trainX, trainX, epochs = epochs, batch_size = mini_batch_size,
                shuffle = True, validation_data=(testX, testX))
# Create encoder
encoder_layer = autoencoder.layers[1](seq_input)
encoder_layer = autoencoder.layers[2](encoder_layer)
encoder = keras.Model(seq_input, encoder_layer)
# Create decoder
encoded_input = keras.Input(shape=(2,))
decoder_layer = autoencoder.layers[-2](encoded_input)
decoder_layer = autoencoder.layers[-1](decoder_layer)
decoder = keras.Model(encoded_input, decoder_layer)

# Save model
autoencoder.save('/Users/Jensaeh/skola/Neural networks/HW3/part2/mse')

# Load model
model = keras.models.load_model('/Users/Jensaeh/skola/Neural'+
                                ' networks/HW3/part2/mse')
encoder_layer_load = model.layers[1](seq_input)
encoder_layer_load = model.layers[2](encoder_layer_load)
encoder = keras.Model(seq_input, encoder_layer_load)

decoder_layer_load = model.layers[3](encoded_input)
decoder_layer_load = model.layers[4](decoder_layer_load)
decoder = keras.Model(encoded_input, decoder_layer_load)

# Make predictions
encoded_data = encoder.predict(testX)
decoded_data = decoder.predict(encoded_data)

#### Plot results autoencoder 1 ####
import matplotlib.pyplot as plt

# Figure 1 (autoencoder1, results)
rows = 2
cols = 10
names = ['\n Original images', '\n Predicted images']
fig, big_axes = plt.subplots(figsize=(20,4), nrows = 2, ncols = 1, sharey=True)
for row, big_ax in enumerate(big_axes, start =1):
    big_ax.set_title(names[row-1], fontsize=12)
    big_ax.axis('off')
    big_ax._frameon = False

for it in range(1,11):
    ind = np.argmax(testY == it-1)
    fig.add_subplot(rows, cols, it)
    plt.imshow(testX[ind].reshape(28,28), cmap = 'gray')
    plt.axis('off')
    fig.add_subplot(rows, cols, it+10)
    plt.imshow(decoded_data[ind].reshape(28,28), cmap = 'gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
fig.set_facecolor('w')
plt.savefig('pred1.png')

# Figure 2 (scatterplot, bottleneck layer)
# 0 1 2 6 7 9 good, 4 5 8 bad
good =[0, 1, 6, 7, 9]
bad = [2, 3, 4, 5, 8]
n = 1000
small_testX = testX[:n]
small_testY = testY[:n]
small_encoded = encoded_data[:n]
fig, ax = plt.subplots(figsize=(10,5))
for it in good:
    ax.scatter(small_encoded[(small_testY == it),0], small_encoded[(small_testY
                                                                     == it),1],
               label=it, s = 10)
ax.legend(title = 'label')
plt.savefig('scatter1.png')

# Observed values for encoder
zeros = [0,15]
ones = [7,0]
sixes = [2,4]
sevens = [6,2]
nines = [2,2]
good_observed = np.array([zeros, ones, sixes, sevens, nines])

# Figure 3 (Feed encoder observed values)
decoded_observed = decoder.predict(good_observed)
good_labels = ['0 (0,15)', '1 (7,0)', '6 (2,4)', '7 (6,2)',
               '9 (2,2)']
fig = plt.figure(figsize=(16,4))
for it in range(5):
    fig.add_subplot(2,4, it+1)
    plt.imshow(decoded_observed[it].reshape(28,28), cmap = 'gray')
    plt.title(good_labels[it])
    plt.axis('off')
plt.show()
plt.savefig('observed1.png')

# Figure 4 (final scatterplot autoencoder 1)
n = 1000
small_testX = testX[:n]
small_testY = testY[:n]
small_encoded = encoded_data[:n]
fig, ax = plt.subplots(figsize=(10,5))
for it in range(10):
    ax.scatter(small_encoded[(small_testY == it),0],
               small_encoded[(small_testY== it),1], label=it, s = 10)
ax.legend(title = 'label')
ax.set_title('Encoded values for the test set (autoencoder1)')
plt.show()
plt.savefig('scatter_final1.png')

#### Autoencoder 2 ####
seq_input = keras.Input(shape=(784,))

# Encoder
layers_encode2 = Dense(50, kernel_initializer='glorot_uniform', activation =
                      'relu')(seq_input)
layers_encode2 = Dense(4, kernel_initializer='glorot_uniform', activation =
                      'relu')(layers_encode2)
# Decoder
layers_decode2 = Dense(784, kernel_initializer='glorot_uniform',
                      activation = 'relu')(layers_encode2)
layers_decode2 = Dense(784)(layers_decode2)

# Autoencoder
autoencoder2 = keras.Model(seq_input, layers_decode2)
autoencoder2.compile(optimizer='adam', loss='mean_squared_error')

# Settings for training
eta = 0.001
mini_batch_size = 8192
epochs = 800 # 800

# Train autoencoder
autoencoder2.fit(trainX, trainX, epochs = epochs, batch_size = mini_batch_size,
                shuffle = True, validation_data=(testX, testX))
# Create encoder
encoder_layer2 = autoencoder2.layers[1](seq_input)
encoder_layer2 = autoencoder2.layers[2](encoder_layer2)
encoder2 = keras.Model(seq_input, encoder_layer2)
# Create decoder
encoded_input2 = keras.Input(shape=(4,))
decoder_layer2 = autoencoder2.layers[-2](encoded_input2)
decoder_layer2 = autoencoder2.layers[-1](decoder_layer2)
decoder2 = keras.Model(encoded_input2, decoder_layer2)

# Save model
autoencoder2.save('/Users/Jensaeh/skola/Neural networks/HW3/part2/mse2')

# Load model
model2 = keras.models.load_model('/Users/Jensaeh/skola/Neural'+
                                ' networks/HW3/part2/mse2')
encoder_layer_load2 = model2.layers[1](seq_input)
encoder_layer_load2 = model2.layers[2](encoder_layer_load2)
encoder2 = keras.Model(seq_input, encoder_layer_load2)

decoder_layer_load2 = model2.layers[3](encoded_input2)
decoder_layer_load2 = model2.layers[4](decoder_layer_load2)
decoder2 = keras.Model(encoded_input2, decoder_layer_load2)

# Make predictions
encoded_data2 = encoder2.predict(testX)
decoded_data2 = decoder2.predict(encoded_data2)

### Plot results autoencoder 2 ###

# Figure 5 (autoencoder2, results)
rows = 2
cols = 10
names = ['\n Original images', '\n Predicted images']
fig, big_axes = plt.subplots(figsize=(20,4), nrows = 2, ncols = 1, sharey=True)
for row, big_ax in enumerate(big_axes, start =1):
    big_ax.set_title(names[row-1], fontsize=12)
    big_ax.axis('off')
    big_ax._frameon = False

for it in range(1,11):
    ind = np.argmax(testY == it-1)
    fig.add_subplot(rows, cols, it)
    plt.imshow(testX[ind].reshape(28,28), cmap = 'gray')
    plt.axis('off')
    fig.add_subplot(rows, cols, it+10)
    plt.imshow(decoded_data2[ind].reshape(28,28), cmap = 'gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
fig.set_facecolor('w')
plt.savefig('pred2.png')

# Inspecting coding rule by taking the mean for each digit for each neuron in
# bottlenck layer
import pandas as pd
bottleneck_data = pd.DataFrame()
for it in range(10):
    bottleneck_data[str(it)] = encoded_data2[testY == it,].mean(axis = 0)
bottleneck_data.head()
bottleneck_data = bottleneck_data.to_numpy()
bottleneck_data = bottleneck_data.transpose()

predicted_inspection = decoder2.predict(bottleneck_data)
good_pred = [0, 1, 4, 6, 7, 9]
titles = ['\n Predicted images', '\n Corresponding bottleneck values']
rows = 2
cols = np.shape(good_pred)[0]
fig, big_axes = plt.subplots(figsize=(20,4), nrows = 2, ncols = 1, sharey=True)
for row, big_ax in enumerate(big_axes, start =1):
    big_ax.set_title(titles[row-1], fontsize=12)
    big_ax.axis('off')
    big_ax._frameon = False

for it in range(1,cols+1):
    ind = good_pred[it-1]
    fig.add_subplot(rows, cols, it)
    plt.imshow(predicted_inspection[ind].reshape(28,28), cmap = 'gray')
    plt.axis('off')
    fig.add_subplot(rows, cols, it+cols)
    plt.bar(x = (0, 5, 10, 15), height = bottleneck_data[ind],
            tick_label=(r'$V_1$', r'$V_2$', r'$V_3$', r'$V_4$'), width = 4,
            color = ['red', 'green', 'blue', 'black'])
    plt.ylim([0,30])
plt.tight_layout()
plt.show()
plt.savefig('inspection_a2.png')

