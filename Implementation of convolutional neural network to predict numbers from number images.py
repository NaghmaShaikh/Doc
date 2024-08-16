from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt

# Download MNIST data and split into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Plot the first image in the dataset
plt.imshow(X_train[0], cmap='gray')
plt.show()

print(X_train[0].shape)

# Reshape the data to include a single channel (grayscale)
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# Convert class vectors to binary class matrices
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

print(Y_train[0])

# Build the model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3)

# Predict on the first 4 images in the test set
print(model.predict(X_test[:4]))

# Actual results for the first 4 images in the test set
print(Y_test[:4])
