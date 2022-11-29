import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyTrigram(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size=100, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        ## TODO: Finish off the method as necessary
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_size)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_size, activation="relu"))
        self.model.add(tf.keras.layers.Dense(self.vocab_size, "softmax"))
        
        ## Second Dense Layer, vocabulary size
        ## Should output probabilities that each word in vocab is next


    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)
        :return: logits: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """

        ## TODO: Implement the method as necessary

        firstwords = self.embedding(inputs[:,0])
        secondwords = self.embedding(inputs[:,1])
        inputs = tf.concat([firstwords, secondwords], 1)
        inputs = self.model(inputs) 

        return inputs

    def generate_sentence(self, word1, word2, length, vocab):
        """
        Given initial 2 words, print out predicted sentence of targeted length.

        :param word1: string, first word
        :param word2: string, second word
        :param length: int, desired sentence length
        :param vocab: dictionary, word to id mapping
        :param model: trained trigram model

        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        output_string = np.zeros((1, length), dtype=np.int)
        output_string[:, :2] = vocab[word1], vocab[word2]

        for end in range(2, length):
            start = end - 2
            output_string[:, end] = np.argmax(self(output_string[:, start:end]), axis=1)
        text = [reverse_vocab[i] for i in list(output_string[0])]

        print(" ".join(text))


#########################################################################################

def get_text_model(vocab):
    '''
    Tell our autograder how to train and test your model!
    '''

    ## TODO: Set up your implementation of the RNN

    ## Optional: Feel free to change or add more arguments!
    model = MyTrigram(len(vocab))

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) 
    
    def perplexityFunction(x, y):
        return tf.math.exp(tf.reduce_sum(loss_metric(x, y)))
    

    acc_metric = perplexityFunction

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate = 0.001), 
        loss=loss_metric, 
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model = model,
        epochs = 1,
        batch_size = 100,
    )


#########################################################################################

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: You might be able to find this somewhere...
    train, test, vocab = get_data("../data/train.txt", "../data/test.txt")

    def process_trigram_data(data):
        X = np.array(data[:-1])
        Y = np.array(data[2:])
        X = np.column_stack((X[:-1], X[1:]))
        return X, Y

    X0, Y0 = process_trigram_data(train)
    X1, Y1 = process_trigram_data(test)


    # TODO: Get your model that you'd like to use
    args = get_text_model(vocab)

    # TODO: Implement get_text_model to return the model that you want to use. 

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    words = 'speak to this brown deep learning student'.split()
    for word1, word2 in zip(words[:-1], words[1:]):
        if word1 not in vocab: print(f"{word1} not in vocabulary")
        if word2 not in vocab: print(f"{word2} not in vocabulary")
        else: args.model.generate_sentence(word1, word2, 20, vocab)

if __name__ == '__main__':
    main()
