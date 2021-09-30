Sentiment Analysis
-------------------

RNNs are used to perform sentiment analysis in keras.

Recurrent Neural Networks (LSTMs), are used here to perform sentiment analysis in Keras. Keras  built-in IMDb movie reviews dataset is used here.

label is an integer (0 for negative, 1 for positive), and the review is stored as a sequence of integers.
To map the integers back to the original words, imdb.get_word_index() can be used.

Here I simply summarized the counts of each word in a document, this representation retains the entire sequence of words (minus punctuation, stopwords, etc.). This is critical for RNNs to function. But, now the features are of different lengths!

In order to feed this data into the RNN, all input documents must have the same length. The maximum review length is limited to max_words by truncating longer reviews and padding shorter reviews with a null value (0). This is done by using the pad_sequences() function in Keras. Here set max_words to 500.

Input is a sequence of words (integer word IDs) of maximum length = max_words, and output is a binary sentiment label (0 or 1).

After the preprocessing, model can be trained using Keras. Model can be compiled by specifying the loss function and optimizer, as well as any evaluation metrics. Specify the approprate parameters, including at least one metric 'accuracy'.

Once compiled, training process can be started. There are two important training parameters - batch size and number of training epochs, which together with model architecture determine the total training time.

