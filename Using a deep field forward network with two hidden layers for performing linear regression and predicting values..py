from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

# Generate the regression dataset
X, Y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)

# Initialize scalers
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()

# Fit and transform the input and output data
scalarX.fit(X)
scalarY.fit(Y.reshape(-1, 1))
X = scalarX.transform(X)
Y = scalarY.transform(Y.reshape(-1, 1))

# Define the model
model = Sequential()
model.add(Input(shape=(2,)))  # Use Input layer to specify input shape
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))  # Use 'linear' activation for regression
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X, Y, epochs=1000, verbose=0)

# Generate new samples for prediction
Xnew, _ = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
Xnew = scalarX.transform(Xnew)

# Make predictions
Ynew = model.predict(Xnew)

# Print predictions
for i in range(len(Xnew)):
    print(f"X={Xnew[i]}, Predicted={Ynew[i][0]}")
