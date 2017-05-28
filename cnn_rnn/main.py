from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LSTM
from keras.layers import Merge
from keras.layers import RepeatVector

VOC_SIZE = 16

model_cnn = Sequential()
model_cnn.add(Conv2D(32, kernel_size=(3, 3), input_shape=[256, 256, 3]))
model_cnn.add(Conv2D(64, kernel_size=(3, 3)))
model_cnn.add(MaxPooling2D())
model_cnn.add(Conv2D(128, kernel_size=(3, 3)))
model_cnn.add(Flatten())
model_cnn.add(Dense(1024))
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(RepeatVector(VOC_SIZE))

model_rnn_1 = Sequential()
model_rnn_1.add(LSTM(128, input_shape=(128, 1), return_sequences=True))
model_rnn_1.add(LSTM(128))

model_rnn_2 = Sequential()
model_rnn_2.add(Merge([model_cnn, model_rnn_1], mode='concat'))
model_rnn_2.add(LSTM(512, return_sequences=True))
model_rnn_2.add(LSTM(512, return_sequences=False))
model_rnn_2.add(Dense(VOC_SIZE, activation='softmax'))

model_rnn_2.compile(optimizer='adam', loss='categorical_crossentropy')
