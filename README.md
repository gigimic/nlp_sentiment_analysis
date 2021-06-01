Sentiment Analysis
-------------------

RNNs are used to perform sentiment analysis in keras.

We use Recurrent Neural Networks, and in particular LSTMs, to perform sentiment analysis in Keras. Keras has a built-in IMDb movie reviews dataset that we can use, with the same vocabulary size.

label is an integer (0 for negative, 1 for positive), and the review is stored as a sequence of integers.
To map the integers back to the original words, weu can use imdb.get_word_index(). 

Unlike the Bag-of-Words approach, where we simply summarized the counts of each word in a document, this representation essentially retains the entire sequence of words (minus punctuation, stopwords, etc.). This is critical for RNNs to function. But it also means that now the features can be of different lengths!

In order to feed this data into the RNN, all input documents must have the same length. Let's limit the maximum review length to max_words by truncating longer reviews and padding shorter reviews with a null value (0). 
This can be done by using the pad_sequences() function in Keras. Here set max_words to 500.

Input is a sequence of words (integer word IDs) of maximum length = max_words, and output is a binary sentiment label (0 or 1).