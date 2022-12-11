import json
import numpy as np
import random
import os
from collections import defaultdict

def preprocess(directory='../data_info/data/', train_test_split=.8, k=None):

<<<<<<< HEAD
    pad = True
=======
    pad = False
>>>>>>> main
    #In ipynb file, use lp as batch size
    #In rnn, mask_zero=True
    '''
    preprocess should return:
        train_data : a 1d list of song ids
        test_data : a 1d list of song ids
        track_to_id : a map of each song name to its id
        relevance : a dictionary {song : [songs in order of relevance]}
    '''
    filepaths = os.listdir(directory)
    if k is not None:
        filepaths = filepaths[:k]

    assert 0 < train_test_split < 1
    assert len(filepaths) > 1
    random.shuffle(filepaths)
    split = int(len(filepaths) * train_test_split)
    train_filepaths = filepaths[:split]
    test_filepaths = filepaths[split:]

    # get train data
    all_train_tracks_name = []

    #Getting size of largest playlist
    longest_playlist = 0
    for filepath in train_filepaths+test_filepaths:
        train_file = open(directory + filepath)
        train_data = json.load(train_file)
        train_playlists = train_data['playlists']
        val = len(train_playlists)
        if val > longest_playlist: 
            longest_playlist = val

    for filepath in train_filepaths:
        train_file = open(directory + filepath)
        train_data = json.load(train_file)
        train_playlists = train_data['playlists']

        for playlist in train_playlists:
            playlist_tracks = playlist['tracks']
            track_names = [x['track_name'] for x in playlist_tracks]
            if pad:
                while len(track_names) < longest_playlist:
                    track_names.append('<PAD>')
<<<<<<< HEAD
                all_train_tracks_name = all_train_tracks_name + ['<PAD>'] + track_names
=======
                all_train_tracks_name = all_train_tracks_name + track_names
>>>>>>> main
            else:
                track_names.append('<BREAK>')
                all_train_tracks_name = all_train_tracks_name + track_names

    # define our vocabulary
    unique_tracks = sorted(set(all_train_tracks_name))
    track_to_id = {name: idx for idx, name in enumerate(unique_tracks)}

    if pad:
        # Swap 0 key with PAD token
        zeroth_song = track_to_id[list(track_to_id.keys())[0]]
        
        track_to_id['<PAD>'], zeroth_song = zeroth_song, track_to_id['<PAD>']
    else:
        # Swap 0 key with PAD token
        zeroth_song = track_to_id[list(track_to_id.keys())[0]]
        
        track_to_id['<BREAK>'], zeroth_song = zeroth_song, track_to_id['<BREAK>']

    id_to_track = {idx: name for idx, name in enumerate(unique_tracks)}

    train_tracks_id = [track_to_id[x] for x in all_train_tracks_name]

    # get test data, removing songs that are not in our vocabulary
    all_test_tracks_name = []

    for filepath in test_filepaths:
        test_file = open(directory + filepath)
        test_data = json.load(test_file)

        test_playlists = test_data['playlists']

        for playlist in test_playlists:
            playlist_tracks = playlist['tracks']
            track_names = [x['track_name'] for x in playlist_tracks]
            # extra line here to ensure that our vocab is constrained to the training data
            track_names = list(filter(lambda x: x in unique_tracks, track_names))
            if pad:
                while len(track_names) < longest_playlist:
                    track_names.append('<PAD>')
<<<<<<< HEAD
                all_test_tracks_name = all_test_tracks_name + ['<PAD>'] + track_names
=======
                all_test_tracks_name = all_test_tracks_name + track_names
>>>>>>> main
            else:
                track_names.append('<BREAK>')
                all_test_tracks_name = all_test_tracks_name + track_names

    test_tracks_id = [track_to_id[x] for x in all_test_tracks_name]


    # calculate relevance
    relevance = defaultdict(lambda: defaultdict(lambda: 0))

    for filepath in filepaths:
        file = open(directory + filepath)
        data = json.load(file)
        playlists = data['playlists']

        for playlist in playlists:
            playlist_tracks = playlist['tracks']
            track_names = [x['track_name'] for x in playlist_tracks]
            track_names = list(filter(lambda x: x in unique_tracks, track_names))
            track_ids = [track_to_id[x] for x in track_names]
            for idx, track1 in enumerate(track_ids):
                for track2 in track_ids[idx:]:
                    relevance[track1][track2] += 1
                    relevance[track2][track1] += 1

    relevance_output = {}

    for song in relevance.keys():
        kv_list = [(k, v) for (k, v) in relevance[song].items()]
        kv_list.sort(key=lambda x: x[1], reverse=True)
        relevance_output[song] = [x[0] for x in kv_list]


    return train_tracks_id, test_tracks_id, track_to_id, relevance_output, longest_playlist+1
<<<<<<< HEAD
=======


def save_data(train, test, track_to_id, relevance, directory='../data_info/saved_preprocessing'):
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)

    np.savetxt(directory + '/train.txt', train, delimiter=',')
    np.savetxt(directory + '/test.txt', test, delimiter=',')
>>>>>>> main

    with open(directory + "/track_to_id.json", "w") as outfile:
        json.dump(track_to_id, outfile)

    with open(directory + "/relevance.json", "w") as outfile:
        json.dump(relevance, outfile)

def get_data(directory, k=None):
    if os.path.exists(directory) and os.path.isdir(directory):
        train = np.genfromtxt(directory + '/train.txt', dtype=np.int32, delimiter=',')
        test = np.genfromtxt(directory + '/test.txt', dtype=np.int32, delimiter=',')

        with open(directory + '/track_to_id.json') as json_file:
            track_to_id = json.load(json_file)

        with open(directory + '/relevance.json') as json_file:
            relevance = json.load(json_file)

        return train, test, track_to_id, relevance

    return preprocess(directory='../data_info/data', k=k)


if __name__ == "__main__":
    # train, test, track, relevance = preprocess(k=100)
    preprocess(train_test_split=.9, k=100)
    # save_data(train, test, track, relevance)
    # get_data('../data_info/saved_preprocessing')

if __name__ == "__main__":
    preprocess(k=2)