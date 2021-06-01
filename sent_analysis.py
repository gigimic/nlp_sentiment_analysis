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
