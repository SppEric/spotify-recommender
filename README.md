# spotify-recommender

## How to run the model
First, ensure the playlist data is present in the `data_info/data/` folder. If not, the Spotify Million Playlist Dataset may be downloaded from `https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge`. Alternatively, you may use the a smaller testing dataset by unzipping `/Users/MasonStang/Desktop/Brown Semester 3/CSCI 1470/Final Project/spotify-recommender/data_info/mpd data 0-1999 for testing.zip`. 

Then, run the model by running all the blocks of code in `code/SpotifyRecommendationModel.ipynb`. After preprocessing and training, you may input a song and receive our model's recommendations of similar songs.

## Results
Overall, our model achieved a R-Precision value of 0.02. While this value is not as high as we would have liked, our model did generate good loss values that decreased with each epoch. Furthermore, our model generated good song recommendations for most inputted songs, especially inputted songs that are popular in that they come up in many playlists.
