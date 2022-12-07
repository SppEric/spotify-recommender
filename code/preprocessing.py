import json
import numpy as np
import random
from collections import defaultdict
import os
import shutil
import csv


def preprocess(filepath='../data_info/data/', k=10000):

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

    # create relevance metrics
    relevance = defaultdict(lambda: defaultdict(int))

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

    # print(train_tracks[:10])
    # print(test_tracks[:10])
    # print(len(train_tracks), len(test_tracks))

    # generate relevance scores
    # relevance maps song id --> every song it appears in a playlist with --> number of times they appear together
    relevance = defaultdict(lambda: defaultdict(lambda: 0))

    for playlist in train_playlists:
        playlist_tracks = playlist['tracks']
        track_names = [x['track_name'] for x in playlist_tracks]
        for idx, track1 in enumerate(track_names):
            id1 = str(track_to_id[track1])
            for track2 in track_names[idx:]:
                id2 = str(track_to_id[track2])
                relevance[id1][id2] += 1
                relevance[id2][id1] += 1


    relevance_output = {}
    for song in relevance.keys():
        kv_list = [(k, v) for (k, v) in relevance[song].items()]
        kv_list.sort(key=lambda x: x[1])
        relevance_output[song] = [x[0] for x in kv_list]

    return train_tracks, test_tracks, track_to_id, relevance_output



def save_data(train, test, track_to_id, relevance, directory='../data_info/saved_preprocessing'):
    if os.path.exists(directory) and os.path.is_dir(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)
    with open('train.csv', 'x') as file:
        writer = csv.writer(file)
        writer.writerows(train)

    with open('test.csv', 'x') as file:
        writer = csv.writer(file)
        writer.writerows(test)

    with open('test.csv', 'x') as file:
        fieldnames=["track", "id"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for key, value in track_to_id.items():
            writer.writerow([key, value])

    with open('relevance.csv', 'x') as file:
        fieldnames=["song", "relevance"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for key, value in relevance.items():
            writer.writerow([key, value])

def get_data


if __name__ == "__main__":
    train, test, track, relevance = preprocess(train_filepath='../data_info/data/',
               test_filepath='../data_info/data/mpd.slice.1000-1999.json', k=2)

    print(train)
    save_data(train, test, track, relevance)
