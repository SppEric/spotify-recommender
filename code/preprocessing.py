import json
import numpy as np
import random
from collections import defaultdict
import os
import shutil
import csv
import json


def preprocess(directory='../data_info/data/', out_dir='../data_info/saved_preprocessing',
               train_test_split=.8, k=None):
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

    train_count = 0
    for filepath in train_filepaths:
        train_file = open(directory + filepath)
        train_data = json.load(train_file)
        train_playlists = train_data['playlists']

        for playlist in train_playlists:
            train_count += 1
            playlist_tracks = playlist['tracks']
            track_names = [x['track_name'] for x in playlist_tracks]
            track_names.append('<BREAK>')
            all_train_tracks_name = all_train_tracks_name + track_names
            if train_count % 1000 == 0:
                print('Currently on train playlist ' + str(train_count))

    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.mkdir(out_dir)


    # define our vocabulary
    unique_tracks = sorted(set(all_train_tracks_name))
    track_to_id = {name: idx for idx, name in enumerate(unique_tracks)}
    with open(out_dir + "/track_to_id.json", "w") as outfile:
        json.dump(track_to_id, outfile)
    # id_to_track = {idx: name for idx, name in enumerate(unique_tracks)}

    train_tracks_id = [track_to_id[x] for x in all_train_tracks_name]
    np.savetxt(out_dir + '/train.txt', np.array(train_tracks_id,dtype=np.int32), delimiter=',')
    print('-------------------------Done train, track_to_id ---------------------')


    # get test data, removing songs that are not in our vocabulary
    all_test_tracks_name = []

    print('beginning test')
    test_counter = 0
    for filepath in test_filepaths:
        test_file = open(directory + filepath)
        test_data = json.load(test_file)

        test_playlists = test_data['playlists']

        for playlist in test_playlists:
            test_counter += 1
            playlist_tracks = playlist['tracks']
            track_names = [x['track_name'] for x in playlist_tracks]
            # extra line here to ensure that our vocab is constrained to the training data
            track_names = list(filter(lambda x: x in unique_tracks, track_names))
            track_names.append('<BREAK>')
            all_test_tracks_name = all_test_tracks_name + track_names
            if test_counter % 1000 == 0:
                print('Currently on test playlist ' + str(test_counter))

    test_tracks_id = [track_to_id[x] for x in all_test_tracks_name]

    print('-------------------------Done with Test ---------------------')
    np.savetxt(out_dir + '/test.txt', np.array(test_tracks_id, dtype=np.int32), delimiter=',')


    # calculate relevance
    relevance = defaultdict(lambda: defaultdict(lambda: 0))

    relevance_counter = 0
    for filepath in filepaths:
        file = open(directory + filepath)
        data = json.load(file)
        playlists = data['playlists']

        for playlist in playlists:
            relevance_counter += 1
            playlist_tracks = playlist['tracks']
            track_names = [x['track_name'] for x in playlist_tracks]
            track_names = list(filter(lambda x: x in unique_tracks, track_names))
            track_ids = [track_to_id[x] for x in track_names]
            for idx, track1 in enumerate(track_ids):
                for track2 in track_ids[idx:]:
                    relevance[track1][track2] += 1
                    relevance[track2][track1] += 1

            if relevance_counter % 1000 == 0:
                print('on playlist for relevance ' + str(relevance_counter))

    relevance_output = {}

    for song in relevance.keys():
        kv_list = [(k, v) for (k, v) in relevance[song].items()]
        kv_list.sort(key=lambda x: x[1])
        relevance_output[song] = [x[0] for x in kv_list]

    with open(out_dir + "/relevance.json", "w") as outfile:
        json.dump(relevance_output, outfile)

    print('Succesfully preprocessed ' + str(k) + ' slices into ' + out_dir)
    # return np.array(train_tracks_id, dtype=np.int32), np.array(test_tracks_id, dtype=np.int32), track_to_id, relevance_output


def save_data(train, test, track_to_id, relevance, directory='../data_info/saved_preprocessing'):
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)

    np.savetxt(directory + '/train.txt', train, delimiter=',')
    np.savetxt(directory + '/test.txt', test, delimiter=',')

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
    preprocess(train_test_split=.8, k=5)
    # save_data(train, test, track, relevance)
    # get_data('../data_info/saved_preprocessing')
