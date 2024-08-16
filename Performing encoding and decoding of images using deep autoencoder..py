import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras.datasets import mnist

# Define encoding dimension
encoding_dim = 32

# Define the input image
input_img = layers.Input(shape=(784,))

# Define the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)

# Define the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# Create the autoencoder model
autoencoder = models.Model(input_img, decoded)

# Create the encoder model
encoder = models.Model(input_img, encoded)

# Define the input for the decoder
encoded_input = layers.Input(shape=(encoding_dim,))

# Retrieve the last layer of the autoencoder model to create the decoder model
decoder_layer = autoencoder.layers[-1]
decoder = models.Model(encoded_input, decoder_layer(encoded_input))

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Load and preprocess the MNIST dataset
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

print(X_train.shape)
print(X_test.shape)

# Train the autoencoder with the training dataset
autoencoder.fit(X_train, X_train,
                 epochs=50,
                 batch_size=256,
                 shuffle=True,
                 validation_data=(X_test, X_test))

# Use the encoder to predict the encoded representations
encoded_imgs = encoder.predict(X_test)

# Use the decoder to reconstruct the images from the encoded representations
decoded_imgs = decoder.predict(encoded_imgs)

# Display the results
n = 10  # Number of digits to display
plt.figure(figsize=(40, 4))

for i in range(n):
    # Display original image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display encoded image
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, 2 * n + i + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
