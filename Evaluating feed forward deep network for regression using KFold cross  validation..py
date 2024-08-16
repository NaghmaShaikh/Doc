import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataframe = pd.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
X = dataset[:, 0:13]
Y = dataset[:, 13]

# Define the model
def create_model():
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Manual Cross-Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
mse_scores = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Standardize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the model
    model = create_model()
    model.fit(X_train, Y_train, epochs=100, batch_size=5, verbose=0)
    
    # Evaluate the model
    mse = model.evaluate(X_test, Y_test, verbose=0)
    mse_scores.append(mse)

# Print results
print(f"Mean Squared Error: {np.mean(mse_scores):.2f} ± {np.std(mse_scores):.2f}")
