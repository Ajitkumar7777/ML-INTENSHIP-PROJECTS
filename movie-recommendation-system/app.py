import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
similarity = cosine_similarity(movie_matrix.T)

def recommend(movie_id):
    scores = list(enumerate(similarity[movie_id]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for i in scores[1:6]:
        print(i)

recommend(1)
