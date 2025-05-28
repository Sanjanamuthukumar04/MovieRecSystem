import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper functions
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    matches = df[df.title.str.lower() == title.lower()]
    if not matches.empty:
        return matches.index[0]
    else:
        return None

# Step 1: Read CSV File
df = pd.read_csv(r"movie_dataset.csv")

# Step 2: Select Features
features = ['keywords', 'cast', 'genres', 'director']

# Step 3: Fill missing values and combine features
for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
    except Exception as e:
        print("Error:", e, "Row:", row)

df["combined_features"] = df.apply(combine_features, axis=1)

# Step 4: Create count matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Step 5: Compute cosine similarity
cosine_sim = cosine_similarity(count_matrix)

# Streamlit layout
col1, col2 = st.columns([3, 6])

with col1:
    st.image("Screenshot.png", use_container_width=True)
    st.image("pcf.jpg", use_container_width=True)

with col2:
    st.title("Welcome to NextWatch!")
    x = st.text_input("Enter the full name of your favourite movie:")

x = x.strip()

# Step 6: Find and recommend similar movies
if x:
    movie_index = get_index_from_title(x)

    if movie_index is None:
        st.write(f"Sorry, the movie '{x}' is not found in the database.")
    else:
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

        st.write(f"Top 30 movies similar to **'{x}'**:")
        i = 0
        for movie in sorted_similar_movies[1:]:
            st.write(get_title_from_index(movie[0]))
            i += 1
            if i >= 30:
                break
