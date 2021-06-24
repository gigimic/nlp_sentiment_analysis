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

nn = 19000
reviews1=[]
# The number of reviews is 25000. so the loop is limited to nn
for ind in range(nn):
    entry=X_train[ind]
    reviews1.append(' '.join([id2word.get(i, " ") for i in entry]))
print('total reviews1  ', len(reviews1))

print('id 2 word for y_train(target) : ', id2word.get(y_train[2]))

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


for ind in range(nn):
    entry=data[ind]
    reviews2.append(" ".join([reverse_index.get(i, "#") for i in entry]))
print('total reviews2  ', len(reviews1))

# print('reviews1 :', reviews1[9],'\n target is...  ', y_train[9],'\n')
# print('reviews2 :', reviews2[10],'\n target is...  ', y_train[10])

# for i in range(305,325):
#     pretty_print_review(i)

from collections import Counter
pos_counts= Counter()
neg_counts= Counter()
total_counts= Counter()

for i in range(nn):
    if(y_train[i] == 0):
        for word in reviews1[i].split(' '):
            pos_counts[word] +=1
            total_counts[word] +=1
    else:
        for word in reviews1[i].split(' '):
            neg_counts[word] +=1
            total_counts[word] +=1


# print(pos_counts.most_common()[:100])

pos_neg_ratios =Counter()

for term, cnt in list(total_counts.most_common()):
    if(cnt > 50):
        pos_neg_ratio = pos_counts[term]/float(neg_counts[term]+1)
        pos_neg_ratios[term] =pos_neg_ratio

for word, ratio in pos_neg_ratios.most_common():
    if(ratio > 1):
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] = -np.log((1 / (ratio+0.01)))

for i in range(50):
    print(pos_neg_ratios.most_common()[i])


print(list(reversed(pos_neg_ratios.most_common()))[0:30])