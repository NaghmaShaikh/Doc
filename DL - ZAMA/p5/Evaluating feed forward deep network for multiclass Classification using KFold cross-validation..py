# Importing libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Loading dataset
df = pd.read_csv('Flower.csv', header=None)
print(df)

# Splitting dataset into input and output variables
X = df.iloc[:, 0:4].astype(float)
y = df.iloc[:, 4]
#print(X)
#print(y)

# Encoding string output into numeric output
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
print(encoded_y)

dummy_Y = np_utils.to_categorical(encoded_y)
print(dummy_Y)

def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.2, random_state=42)

# Create and fit the model
estimator = baseline_model()
estimator.fit(X_train, Y_train, epochs=100, shuffle=True, validation_data=(X_test, Y_test))

# Predicting
action = estimator.predict(X_test)

for i in range(25):
    print(Y_test[i])
print('^^^^^^^^^^^^^^^^^^^^^^')
for i in range(25):
    print(action[i])
