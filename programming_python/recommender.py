import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

ratings = pd.read_csv("programming_python/ratings.csv", header=0)
ratings = ratings.drop(columns=["timestamp"])

movies = pd.read_csv("programming_python/movies.csv", header=0)
movies = movies.drop(columns="genres")

user_item_matrix = ratings.pivot(
    index="userId", columns="movieId", values="rating"
).fillna(0)

R = user_item_matrix.values

user_means = np.mean(R, axis=1)

R_demeaned = R - user_means.reshape(-1, 1)

U, S, Vh = svds(R_demeaned, k=100)
Sigma = np.diag(S)

prediction_matrix = U @ Sigma @ Vh + user_means.reshape(-1, 1)


def recommender(R, prediction_matrix, num_recom, user):
    user_idx = user - 1
    sorted_predictions = np.argsort(-prediction_matrix[user_idx])
    unwatched_indices = np.where(R[user_idx] == 0)[0]
    recommended_movies_indices = [
        int(movie) for movie in sorted_predictions if movie in unwatched_indices
    ][:num_recom]
    recommended_movies_ids = [
        int(user_item_matrix.columns[idx]) for idx in recommended_movies_indices
    ]
    recommended_movies = [
        movies.loc[movies["movieId"] == id, "title"].iloc[0]
        for id in recommended_movies_ids
    ]
    return recommended_movies


num = int(input("Enter number of wanted recommendations: "))
use = int(input("Enter UserId: "))
print(
    "The top",
    num,
    "recommendations for User",
    use,
    "are",
    recommender(R, prediction_matrix, num, use),
)
