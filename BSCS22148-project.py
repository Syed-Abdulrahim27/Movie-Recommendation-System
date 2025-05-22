import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


df = pd.read_csv('./expanded_movie_dataset.csv')

movieCols = ["Movie 1", "Movie 2", "Movie 3", "Movie 4", "Movie 5"]

for col in movieCols:
    df[col] = df[col].astype(str).str.strip().str.lower()

fixes = {
    "interstellars": "interstellar",
    "top gun": "top gun maverick",
    "muna bhi mbbs": "munna bhai mbbs",
    "bhjrngi bhai jan": "bajrangi bhaijaan",
    "money hiest korean": "money heist",
    "mission impossible series": "mission impossible",
    "the heirs": "heirs",
    "matrix": "the matrix",
    "cars": "cars (2006)",
    "tron": "tron legacy",
}

df[movieCols] = df[movieCols].map(lambda title: fixes.get(title, title))

userMovies = {}
for i, row in df.iterrows():
    name = row['Name'].strip().lower()
    movies = [row[col] for col in movieCols if pd.notna(row[col])]
    userMovies[name] = list(set(movies))

unique_movies = set()
for movie_list in userMovies.values():
    unique_movies.update(movie_list)

all_movies = sorted(unique_movies)

print(all_movies)
print(len(all_movies))

print("Sample cleaned data:\n")
for user, movies in list(userMovies.items()):
    print(f"{user}: {movies}")


userMovieMatrix_df = pd.DataFrame(0, index=userMovies.keys(), columns=all_movies)

for user, movies in userMovies.items():
    for movie in movies:
        userMovieMatrix_df.loc[user, movie] = 1

print(userMovieMatrix_df)
def euclideanDistance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def assignClusters(centroids, X):
    clusters = {i: [] for i in range(len(centroids))}
    clusterLabels = np.zeros(len(X), dtype=int)

    for index, point in enumerate(X):
        distances = [euclideanDistance(point, centroid) for centroid in centroids]
        closestCentroid = np.argmin(distances)
        clusters[closestCentroid].append(point)
        clusterLabels[index] = closestCentroid
    return clusters, clusterLabels

def updateCentroids(clusters, centroids):
    newCentroids = []
    for i in range(len(clusters)):
        if clusters[i]:
            newCentroids.append(np.mean(clusters[i], axis=0))
        else:
            newCentroids.append(centroids[i])
    return np.array(newCentroids)

X = userMovieMatrix_df.drop(columns='Cluster', errors='ignore').values

k = 5
max_iters = 100
tolerance = 1e-4

random.seed(42) 
centroids = np.array(random.sample(list(X), k))

for iteration in range(max_iters):
    clusters, clusterLabels = assignClusters(centroids, X)
    newCentroids = updateCentroids(clusters, centroids)

    if np.max(np.linalg.norm(newCentroids - centroids, axis=1)) < tolerance:
        print(f"Converged in {iteration + 1} iterations.")
        break

    centroids = newCentroids

userMovieMatrix_df['Cluster'] = clusterLabels

userMovieMatrix_df[['Cluster']].to_csv("expanded_movie_dataset_clusters.csv")



def recommendMovies(userName, userMovies, userMovieMatrix_df, top_n=5):
    userName = userName.lower().strip()
    if userName not in userMovieMatrix_df.index:
        return f"{userName} not found in dataset."

    clusterId = userMovieMatrix_df.loc[userName, 'Cluster']

    sameClusterUsers = userMovieMatrix_df[userMovieMatrix_df['Cluster'] == clusterId].index.tolist()

    userAlreadySeen = set(userMovies[userName])

    movieCounts = {}
    for otherUser in sameClusterUsers:
        if otherUser == userName:
            continue
        for movie in userMovies[otherUser]:
            if movie not in userAlreadySeen:
                movieCounts[movie] = movieCounts.get(movie, 0) + 1

    recommended = sorted(movieCounts.items(), key=lambda x: x[1], reverse=True)

    return [movie for movie, count in recommended[:top_n]]

#print("Recommended Movies:",recommendMovies("syed abdul rahim", userMovies, userMovieMatrix_df))
print("Recommended Movies:",recommendMovies("ebad junaid", userMovies, userMovieMatrix_df))

def cosineSimilarityCustom(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)


features_df = userMovieMatrix_df.drop(columns='Cluster')

n_users = features_df.shape[0]
cosine_sim_matrix = np.zeros((n_users, n_users))

for i in range(n_users):
    for j in range(n_users):
        cosine_sim_matrix[i][j] = cosineSimilarityCustom(features_df.iloc[i], features_df.iloc[j])

plt.figure(figsize=(12, 10))

sns.heatmap(
    cosine_sim_matrix,
    xticklabels=userMovieMatrix_df.index.tolist(),
    yticklabels=userMovieMatrix_df.index.tolist(),
    cmap='viridis',
    annot=False
)

plt.title("User-User Similarity (Custom Cosine)")
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.show()

userMovieMatrix_df['Cluster'].value_counts().sort_index().plot(kind='bar')
plt.title("Number of Users per Cluster")
plt.xlabel("Cluster")
plt.ylabel("User Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
























# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X)
# centroids_reduced = pca.transform(centroids)

# plt.figure(figsize=(8, 6))

# colors = plt.cm.get_cmap('tab10', k)

# for i in range(k):
#     clusterPoints = X_reduced[clusterLabels == i]
#     plt.scatter(clusterPoints[:, 0], clusterPoints[:, 1], color=colors(i), label=f"Cluster {i}")
#     plt.scatter(centroids_reduced[i, 0], centroids_reduced[i, 1],
#                 color=colors(i), marker='X', s=200, edgecolor='black', linewidth=1.5)

# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("K-Means Clustering of Movie/User Data")
# plt.legend()
# plt.show()

# userMovieMatrix_df['Cluster'].value_counts().sort_index().plot(kind='bar')
# plt.title("Number of Users per Cluster")
# plt.xlabel("Cluster")
# plt.ylabel("User Count")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()






