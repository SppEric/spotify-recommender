import tensorflow as tf
import numpy as np
from preprocessing import preprocess
from types import SimpleNamespace



class Model(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=256, embed_size=300):
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


        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_size, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(self.embed_size, return_sequences=True, return_state=False)
        self.model = tf.keras.Sequential(
            [       
                tf.keras.layers.Dense(10 * self.embed_size, activation='relu'),
                tf.keras.layers.Dense(6 * self.embed_size, activation='relu'),
                tf.keras.layers.Dense(self.vocab_size, activation='softmax')
            ]
        )

    def call(self, inputs):
        x = self.embedding(inputs)  
        x = self.lstm(x)
        
        # Send to sequence
        x = self.model(x)
        return x

    ##########################################################################################

    def generate_recommendations(self, word1, length, vocab):
        """
        Takes a model, vocab, selects from the most likely next song from the model's distribution
        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}

        first_string = word1
        first_word_index = vocab[word1]
        next_input = np.array(first_word_index).reshape((1,1))
        text = [first_string]

        # Find top songs based off the logits
        logits = self.call(next_input)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-length:]
        text = [reverse_vocab[n] for n in top_n]

        return text


#########################################################################################

def get_text_model(vocab, relevance):

    model = Model(len(vocab))

    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy()

    # Initialize then call RPrecision metric
    def RPrecision(predictions, labels):
        PAD_TOKEN = 0

        # Set up prediction array
        prediction_arr = predictions.numpy().flatten().astype(int)
        input_song = labels.numpy().flatten().astype(int)[0]
        #print(prediction_arr)
        prediction_arr = prediction_arr[prediction_arr != PAD_TOKEN]

        predict_set = set(prediction_arr)
        relevant_songs = relevance[input_song]
        relevant_songs = relevant_songs[:len(prediction_arr)]
        ground_truth = set(relevant_songs)

        # Return mean of running total to get running mean
        return len(predict_set.intersection(ground_truth)) / len(ground_truth)


    acc_metric = RPrecision

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.005), 
        loss=loss_metric, 
        metrics=[acc_metric],
        run_eagerly=True
    )

    return SimpleNamespace(
        model = model,
        epochs = 10,
        batch_size = 100,
    )

#########################################################################################

def main():
    train_id, test_id, vocab, relevance, lp = preprocess(directory='../data_info/data/', train_test_split=0.8, k=3)

    train_id = np.array(train_id)
    test_id  = np.array(test_id)    

    # Training and validation are aligned because we require the input song for RPrecision
    X0, Y0 = train_id, train_id
    X1, Y1 = test_id,  test_id

    args = get_text_model(vocab, relevance)

    data = args.model.fit(
        X0, Y0,
        epochs=2, 
        batch_size=lp,
        validation_data=(X1, Y1)
    )

    def RPrecision(predictions, labels):
        PAD_TOKEN = 0
        predict_set = set(predictions)
        labels = labels[:len(predict_set)]
        
        ground_truth = set(labels)

        return len(predict_set.intersection(ground_truth)) / len(ground_truth)

    for word1 in ['Closer']:
        if word1 not in vocab: print(f"{word1} not in vocabulary")            
        else: print(args.model.generate_recommendations(word1, 10, vocab))
        print()

    ids = relevance[vocab['Closer']]
    id_to_track = {id: name for name, id in vocab.items()}
    tracks =[id_to_track[id] for id in ids]
    print(tracks[:30])
    print()
    print("R-Precision: " + str(RPrecision(args.model.generate_recommendations(word1, 10, vocab), [id_to_track[x] for x in relevance[vocab['Closer']]])))

if __name__ == '__main__':
    main()
