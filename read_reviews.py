from os import listdir
from collections import Counter

from utils_prog import clean_text_for_comparison
from utils_prog import load_doc

dir_pos = '...../data_imdb/txt_sentoken/pos/'
dir_neg = '...../data_imdb/txt_sentoken/neg/'


review_pos = []
labels_pos = []
vocab_pos = []
pos_words_count = Counter()
for files in listdir(dir_pos):
    if files.endswith("txt"):
        # read the review
        filename = dir_pos + files 
        # print(filename)
        review = load_doc(filename)
        # clean the text
        # review = clean_text_for_comparison(text)
        tokens = [word for word in review.split(' ')]
        for word in tokens:
            vocab_pos.append(word)
            pos_words_count[word] += 1
        review_pos.append(review)
        labels_pos.append('positive')
        

print('number of reviews : ',len(review_pos))
# print(review_pos[3])
# print(labels_pos[3])
print('no of words : ', len(vocab_pos))
print(vocab_pos[3])
print('most common : ', pos_words_count.most_common(50))

review_neg = []
labels_neg = []
vocab_neg = []
neg_words_count = Counter()
for files in listdir(dir_neg):
    if files.endswith("txt"):
        # read the review
        filename=dir_neg + files 
        # print(filename)
        review = load_doc(filename)
        # clean the text
        # review = clean_text_for_comparison(text)
        tokens = [word for word in review.split(' ')]
        for word in tokens:
            vocab_neg.append(word)
            neg_words_count[word] += 1
        review_neg.append(review)
        labels_neg.append('negative')
        
print('number of neg reviews : ',len(review_neg))
print(review_neg[3])
print(labels_neg[3])
print('no of words in neg reviews : ', len(vocab_neg))
print(vocab_neg[3])
print('most common : ', neg_words_count.most_common(50))

import numpy as np 
# generate 5 random integers between 0 and 100
x = np.random.randint(low = 40, high=100, size = (5))
print(x)

reviews_all = []
labels_all =[]
for i in range(len(review_neg)):
    reviews_all.append(review_neg[i])
    labels_all.append(labels_neg[i])
    reviews_all.append(review_pos[i])
    labels_all.append(labels_pos[i]) 

print('review..', reviews_all[11])
print('label.. ', labels_all[11])
