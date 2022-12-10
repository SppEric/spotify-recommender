import tensorflow as tf
import numpy as np
from accuracy import RPrecision
from preprocessing import preprocess
from types import SimpleNamespace

window_size = 20

class MyRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=128, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size

        ## TODO: Finish off the method as necessary.
        ## Define an embedding component to embed the word indices into a trainable embedding space.
        ## Define a recurrent component to reason with the sequence of data. 
        ## You may also want a dense layer near the end...    
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_size, mask_zero=True) # add mask_zero = true
        self.lstm = tf.keras.layers.LSTM(self.embed_size, return_sequences=True, return_state=False)
        self.model = tf.keras.Sequential(
            [       
                tf.keras.layers.Dense(10 * self.embed_size, activation='relu'),
                tf.keras.layers.Dense(self.vocab_size, activation='softmax')
            ]
        )

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.
        """
        ## TODO: Implement the method as necessary
        x = self.embedding(inputs)
        x = self.lstm(x)
        
        # Send to sequence
        x = self.model(x)
        return x

    ##########################################################################################

    def generate_sentence(self, word1, length, vocab, sample_n=10):
        """
        Takes a model, vocab, selects from the most likely next word from the model's distribution
        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}

        first_string = word1
        first_word_index = vocab[word1]
        next_input = np.array(first_word_index).reshape((1,1))
        text = [first_string]

        for i in range(length):
            logits = self.call(next_input)
            logits = np.array(logits[0,0,:])
            top_n = np.argsort(logits)[-sample_n:]
            n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n,p=n_logits)

            text.append(reverse_vocab[out_index])
            next_input = np.array(out_index).reshape((1,1))

        print(", ".join(text))


#########################################################################################

def get_text_model(vocab):
    '''
    Tell our autograder how to train and test your model!
    '''

    ## TODO: Set up your implementation of the RNN

    ## Optional: Feel free to change or add more arguments!
    model = MyRNN(len(vocab))

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_metric  = RPrecision()

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.006), 
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
    ##   HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    ##   from train_x and test_x. You also need to drop the first element from train_y and test_y.
    ##   If you don't do this, you will see very, very small perplexities.
    ##   HINT: You might be able to find this somewhere...
    train_id, test_id, vocab, relevance, lp = preprocess("../data/train.txt", "../data/test.txt")

    train_id = np.array(train_id)
    test_id  = np.array(test_id)    
    X0, Y0 = train_id[:-1], train_id[1:]
    X1, Y1 = test_id[:-1],  test_id[1:]

    # # Reshape training data into window sized batches
    # X0, Y0 = X0[:-(len(X0) % window_size)], Y0[:-(len(Y0) % window_size)]
    # X0 = X0.reshape(-1, 20)
    # Y0 = Y0.reshape(-1, 20)

    # # Reshape test data into window sized batches
    # X1, Y1 = X1[:-(len(X1) % window_size)], Y1[:-(len(Y1) % window_size)]
    # X1 = X1.reshape(-1, 20)
    # Y1 = Y1.reshape(-1, 20)

    ## TODO: Get your model that you'd like to use
    args = get_text_model(vocab)

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=lp,
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    for word1 in 'speak to this brown deep learning student'.split():
        if word1 not in vocab: print(f"{word1} not in vocabulary")            
        else: args.model.generate_sentence(word1, 20, vocab, 10)

if __name__ == '__main__':
    main()
