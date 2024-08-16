from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

dataset = loadtxt('D:/STUDY/MSC IT - II/SEM 4 Journal/DL/p3/pima-indians-diabetes.data.csv', delimiter=',')

X = dataset[:, 0:8]
Y = dataset[:, 8]

model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
l
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=10, verbose=1)

loss, accuracy = model.evaluate(X, Y)
print(f"Accuracy of model is {accuracy * 100:.2f}%")

predictions = model.predict(X, batch_size=4)

for i in range(5):
    print(f"Input: {X[i].tolist()}, Prediction: {predictions[i][0]:.4f}, Actual: {Y[i]}")

    
