from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

model = Sequential()
model.add(Input(shape=(2,)))  
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(X, Y, epochs=500, verbose=0)  

Xnew, Yreal = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)

Ynew = (model.predict(Xnew) > 0.5).astype(int)

for i in range(len(Xnew)):
    print(f"X={Xnew[i]}, Predicted={Ynew[i][0]}, Desired={Yreal[i]}")
