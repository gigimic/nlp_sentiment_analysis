from keras.datasets import imdb
# import tensorflow as tf
# print('done version tf ', tf.__version__)

# Set the vocabulary size
vocabulary_size = 5000

# Load in training and test data (note the difference in convention compared to scikit-learn)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print("Loaded dataset with {} training samples, {} test samples".format(len(X_train), len(X_test)))

# Inspect a sample review and its label
# label is an integer (0 for negative, 1 for positive), 
# print("--- Review ---")
# print(X_train[7])
print("--- Label ---")
print(y_train[7])

# These are word IDs that have been preassigned to individual words. 
# To map them back to the original words, we can use the dictionary returned by imdb.get_word_index().

# Map word IDs back to words
word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print("--- Review (with words) ---")
# print([id2word.get(i, " ") for i in X_train[7]])
print("--- Label ---")
print(y_train[7])

# Pad sequences: In order to feed this data into the RNN, all input documents must have the same length. 
# So we can  limit the maximum review length to max_words by truncating longer reviews and 
# padding shorter reviews with a null value (0). We can do this using the pad_sequences() function in Keras. 
# Here set max_words to 500.

from keras.preprocessing import sequence

# Set the maximum number of words per document (for both training and testing)
max_words = 500

# Pad sequences in X_train and X_test
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# Design an RNN model for sentiment analysis¶

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# Design the model
embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# Compile model, specifying a loss function, optimizer, and metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Once compiled, start the training process. There are two important 
# training parameters - batch size and number of training epochs, 
# which together with model architecture determine the total training time.

# Specify training parameters: batch size and number of epochs
batch_size = 64
num_epochs = 3

# (optional): Reserve/specify some training data for validation (not to be used for training)
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]  # first batch_size samples
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]  # rest for training

# Train the model
model.fit(X_train2, y_train2,
          validation_data=(X_valid, y_valid),
          batch_size=batch_size, epochs=num_epochs)

import os 
# Save the model, so that we can quickly load it in future (and perhaps resume training)
model_file = "rnn_model.h5"  # HDF5 file
model.save(os.path.join(temp_dir, model_file))

# Later it can be loaded using keras.models.load_model()
#from keras.models import load_model
#model = load_model(os.path.join(temp_dir, model_file))

# After training the model, check to see how well it performs on unseen test data.

# Evaluate the model on the test set
scores = model.evaluate(X_test, y_test, verbose=0)  # returns loss and other metrics specified in model.compile()
print("Test accuracy:", scores[1])  # scores[1] should correspond to accuracy if you passed in metrics=['accuracy']
