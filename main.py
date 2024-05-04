# %%
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical

# %%
text = (open("sonnets.txt").read())
text = text.lower().split()
text.__len__()

# %%
chars = sorted(list(set(text)))
numToChar = {n:word for n, word in enumerate(chars)}
charToNum = {word:n for n, word in enumerate(chars)}

# %%
x = []
y = []
length = len(text)
lengthSeq = 100

for i in range(0, length - lengthSeq, 1):
    seq = text[i:i + lengthSeq]
    label = text[i + lengthSeq]
    x.append([charToNum[word] for word in seq])
    y.append(charToNum[label])

# %%
xMOD = np.reshape(x, (len(x), lengthSeq, 1))
xMOD = xMOD / float(len(chars))
yMOD = to_categorical(y)

# %%
model = Sequential()
model.add(LSTM(400, input_shape=(xMOD.shape[1], xMOD.shape[2]), return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(400))
model.add(Dropout(.2))
model.add(Dense(yMOD.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.fit(xMOD, yMOD, epochs=10, batch_size=200, verbose=2)

# %%
stringMapped = x[5]
for i in range(lengthSeq):
    x = np.reshape(stringMapped,(1, len(stringMapped), 1))
    x = x / float(len(chars))
    predictIndex = np.argmax(model.predict(x, verbose=0))
    sequence = [numToChar[value] for value in stringMapped]
    stringMapped.append(predictIndex)
    stringMapped = stringMapped[1:len(stringMapped)]

# %%
" ".join(seq)


