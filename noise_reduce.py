from keras.datasets import imdb
import numpy as np 
# Set the vocabulary size
vocabulary_size = 10000


def pretty_print_review(i):
    print(y_train[i], ' .. : ', reviews1[i][:160])

# here the reviews are given as word ids. we need to convert the ids to words to read the review
# Load in training and test data (note the difference in convention compared to scikit-learn)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print("Loaded dataset with {} training samples, {} test samples".format(len(X_train), len(X_test)))

# This is method-1 of reading the reviews
word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print("--- Review (with words) ---")
print(' '.join([id2word.get(i, " ") for i in X_train[0]]))
print("--- Label ---  ", y_train[0])


reviews1=[]
# The number of reviews is 25000. so the loop is limited to 500
for ind in range(500):
    entry=X_train[ind]
    reviews1.append(' '.join([id2word.get(i, " ") for i in entry]))
print('total reviews1  ', len(reviews1))

print('id 2 word for 1 : ', id2word.get(y_train[10]))
print('id 2 word for 1 : ', id2word.get(1))


print('id 2 word for 0 : ', id2word.get(0))


# This is method - 2 of reading reviews

data = np.concatenate((X_train, y_train), axis=0)
targets = np.concatenate((X_test, y_test), axis=0)
# data contains all the reviews and targest are the sentiments (positive or negative)
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()]) 
# decoded = " ".join( [reverse_index.get(i-3, "#") for i in data[0]] )
# print(decoded)

reviews2=[]
# for ind, entry in data:
#     reviews2.append(" ".join([reverse_index.get(i, "#") for i in entry]))


for ind in range(500):
    entry=data[ind]
    reviews2.append(" ".join([reverse_index.get(i, "#") for i in entry]))
print('total reviews2  ', len(reviews1))

print('reviews1 :', reviews1[9],'\n target is...  ', y_train[9],'\n')
print('reviews2 :', reviews2[10],'\n target is...  ', y_train[10])

for i in range(305,325):
    pretty_print_review(i)
    # pretty_print_review(105)
    # pretty_print_review(108)
    # pretty_print_review(201)