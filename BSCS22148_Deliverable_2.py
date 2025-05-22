import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
# 1. Load CSV
df = pd.read_csv('./dataset.csv')

# 2. Combine movie columns into a single list per user
movie_cols = ["Movie 1", "Movie 2", "Movie 3", "Movie 4", "Movie 5"]

# Normalize movies: lowercase, strip spaces
for col in movie_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# 3. Fixing typos from dataset
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

def fix_title(title):
    return fixes.get(title, title)

df[movie_cols] = df[movie_cols].applymap(fix_title)

# 4. Create a dict: user -> list of movies
user_movies = {}
for i, row in df.iterrows():
    name = row['Name'].strip().lower()
    movies = [row[col] for col in movie_cols if pd.notna(row[col])]
    user_movies[name] = list(set(movies))  # remove duplicates per user


all_movies = sorted({movie for movies in user_movies.values() for movie in movies})

# Output cleaned user-movie dictionary
print("Sample cleaned data:\n")
for user, movies in list(user_movies.items()):
    print(f"{user}: {movies}")



# Step 2: Create the user-movie matrix
user_movie_df = pd.DataFrame(0, index=user_movies.keys(), columns=all_movies)

for user, movies in user_movies.items():
    for movie in movies:
        user_movie_df.loc[user, movie] = 1

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(user_movie_df)
# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, xticklabels=user_movie_df.index, yticklabels=user_movie_df.index, cmap='viridis', annot=True, fmt=".2f")
plt.title("User-User Similarity (Cosine)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()