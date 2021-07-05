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
        

print('number of positive reviews : ',len(review_pos))
print('no of words : ', len(vocab_pos))
print('most common words in positive reviews : ', pos_words_count.most_common(20))

review_neg = []
labels_neg = []
vocab_neg = []
neg_words_count = Counter()

for files in listdir(dir_neg):
    if files.endswith("txt"):
        # read the review
        filename=dir_neg + files 
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
        
print('number of negative reviews : ',len(review_neg))
print('no of words in neg reviews : ', len(vocab_neg))
print('most common words in negative reviews : ', neg_words_count.most_common(20))

# print('most common total : ', vocab_total.most_common(50))

reviews_all = []
labels_all =[]
# mixing the negative and positive reviews
for i in range(len(review_neg)):
    reviews_all.append(review_neg[i])
    labels_all.append(labels_neg[i])
    reviews_all.append(review_pos[i])
    labels_all.append(labels_pos[i]) 

print('total no of reviews... labels... ', len(reviews_all), len(labels_all))

def pretty_print_review(i):
    print(labels_all[i], ' .. : ', reviews_all[i][:100])

# for i in range(305,325):
#     pretty_print_review(i)

# as there are many words common in both positive and negative reviews 
# we use a method to find the positive to negative ratio

import numpy as np

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

print('words with higher pos/neg ratio........ ')
print(pos_neg_ratios.most_common(20))
print('words with higher neg/pos ratio........ ')
print(list(reversed(pos_neg_ratios.most_common()))[0:20])

# generating word to index and index to word    
word2index={}
index2word = {}
num_words=1
for word, i in vocab_total.items():
    if word not in word2index:
        word2index[word]=num_words
        index2word[num_words] = word
        num_words += 1

# print(len(word2index))
# print(index2word[15])
# print(word2index['viewed'])
# print(word2index['awful'])

# input data for analysis - first layer
layer_0 = np.zeros((1, len(word2index)))

def update_input_layer (review):
    global layer_0
    layer_0 *= 0
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1

update_input_layer(reviews_all[0])
print(len(layer_0[0]))
# print([layer_0[0][i] for i in layer_0[0][i]>0])
print(np.max(layer_0[0]))
print(np.argmax(layer_0[0]))
nn1 = np.argmax(layer_0[0])
print(index2word[nn1])