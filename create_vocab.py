import numpy as np
from collections import Counter

text = '''Bharat Biotech is yet to begin supply for a batch of vaccines, an order for which was placed by 
the Health Ministry in May, according to an affidavit filed by the Ministry in the Supreme Court on Saturday.
The Health Ministry told the apex court that it expected to receive at least 51 crore doses from January-July. 
Thirty two crore of these have already been administered as of Sunday evening.
'''
# print(text)
 

words_index={}
words_count =Counter()
for i, word in enumerate(text.split(' ')):
    words_index[word] = i
    words_count[word] += 1

print(words_index)
print(words_count.most_common())

vocab_size = len(words_count) 
print(vocab_size)
layer_0 = np.zeros((1, vocab_size))
print(layer_0)

for index, key  in words_index.items():
    print(index, key)
    
