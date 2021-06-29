from os import listdir
from collections import Counter

from utils_prog import clean_text_for_comparison
from utils_prog import load_doc

# data from
# https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/

dir_pos = '/home/gigi/udacityNLP/projects/data_imdb/txt_sentoken/pos/'
dir_neg = '/home/gigi/udacityNLP/projects/data_imdb/txt_sentoken/neg/'


vocab_total = Counter()

review_pos = []
labels_pos = []
vocab_pos = []
pos_words_count = Counter()
for files in listdir(dir_pos):
    if files.endswith("txt"):
        # read the review
        filename = dir_pos + files 
        # print(filename)
        text = load_doc(filename)
        # clean the text
        review = clean_text_for_comparison(text)
        tokens = [word for word in review.split(' ')]
        for word in tokens:
            vocab_pos.append(word)
            pos_words_count[word] += 1
            vocab_total[word] += 1
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
        text = load_doc(filename)
        # clean the text
        review = clean_text_for_comparison(text)
        tokens = [word for word in review.split(' ')]
        for word in tokens:
            vocab_neg.append(word)
            neg_words_count[word] += 1
            vocab_total[word] += 1
        review_neg.append(review)
        labels_neg.append('negative')
        
print('number of neg reviews : ',len(review_neg))
# print(review_neg[3])
# print(labels_neg[3])
print('no of words in neg reviews : ', len(vocab_neg))
print(vocab_neg[3])
print('most common negative : ', neg_words_count.most_common(50))

print('most common total : ', vocab_total.most_common(50))

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

# print('review..', reviews_all[10])
# print('label.. ', labels_all[10])
print('no of reviews... labels... ', len(reviews_all), len(labels_all))

def pretty_print_review(i):
    print(labels_all[i], ' .. : ', reviews_all[i][:100])

# for i in range(305,325):
#     pretty_print_review(i)

pos_neg_ratios = Counter()
for term, cnt in list(vocab_total.most_common()):
    if(cnt > 50):
        pos_neg_ratio = pos_words_count[term]/(float(neg_words_count[term]) + 1)
        pos_neg_ratios[term] = pos_neg_ratio

for word, ratio in pos_neg_ratios.most_common():
    if(ratio >1):
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] = -np.log(1/(ratio+0.01))

print(pos_neg_ratios.most_common(50))
print(list(reversed(pos_neg_ratios.most_common()))[0:30])
    
word2index={}
num_words=1
for word, i in vocab_total.items():
    if word not in word2index:
        word2index[word]=num_words
        num_words += 1
        if (num_words < 20):
            print(word, word2index[word])

print(len(word2index))
