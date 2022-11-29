import json
import numpy as np
import random

def preprocess(train_filepath='data_info/data/', test_filepath='data_info/data/', k=10000):

    '''
    preprocess should return:
        train_data : a 1d list of song ids
        test_data : a 1d list of song ids
        track_to_id : a map of each song name to its id
    '''

    # get train data
    train_file = open(train_filepath)
    train_data = json.load(train_file)

    train_playlists = train_data['playlists']
    all_train_tracks = []

    for playlist in train_playlists:
        playlist_tracks = playlist['tracks']
        track_names = [x['track_name'] for x in playlist_tracks]
        # track_names.append('<BREAK>')
        all_train_tracks = all_train_tracks + track_names

    unique_tracks = sorted(set(all_train_tracks))
    track_to_id = {name: idx for idx, name in enumerate(unique_tracks)}
    id_to_track = {idx: name for idx, name in enumerate(unique_tracks)}

    train_tracks = [track_to_id[x] for x in all_train_tracks]

    # get test data, removing songs that are not in our vocabulary
    test_file = open(test_filepath)
    test_data = json.load(test_file)

    test_playlists = test_data['playlists']
    all_test_tracks = []

    for playlist in test_playlists:
        playlist_tracks = playlist['tracks']
        track_names = [x['track_name'] for x in playlist_tracks]
        # extra line here to ensure that our vocab is constrained to the training data
        track_names = list(filter(lambda x: x in unique_tracks, track_names))
        # track_names.append('<BREAK>')
        all_test_tracks = all_test_tracks + track_names

    test_tracks = [track_to_id[x] for x in all_test_tracks]

    print(train_tracks[:10])
    print(test_tracks[:10])
    return train_tracks, test_tracks, track_to_id




def randomly_sample(inputs, k):
    return random.choices(inputs, k=k)


if __name__ == "__main__":
    preprocess(train_filepath='../data_info/data/mpd.slice.0-999.json', test_filepath='../data_info/data/mpd.slice.1000-1999.json')