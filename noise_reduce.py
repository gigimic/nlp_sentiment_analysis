from keras.datasets import imdb
import numpy as np 
# Set the vocabulary size
vocabulary_size = 5000

# Load in training and test data (note the difference in convention compared to scikit-learn)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print("Loaded dataset with {} training samples, {} test samples".format(len(X_train), len(X_test)))

# word2id = imdb.get_word_index()
# id2word = {i: word for word, i in word2id.items()}
# print("--- Review (with words) ---")
# print([id2word.get(i, " ") for i in X_train[7]])
# print("--- Label ---")
# print(y_train[7])

data = np.concatenate((X_train, y_train), axis=0)
targets = np.concatenate((X_test, y_test), axis=0)
print(len(data[0]))
print(len(data))
# print(data[0])

index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()]) 
decoded = " ".join( [reverse_index.get(i-3, "#") for i in data[0]] )
print(decoded)