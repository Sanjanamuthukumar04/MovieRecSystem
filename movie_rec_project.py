import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#this imports a class. so you should initialize it
from sklearn.metrics.pairwise import cosine_similarity
#this imports a method. no need to initialize

# Helper functions
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    matches = df[df.title.str.lower() == title.lower()]
    if not matches.empty:
        return matches.index[0]
    else:
        return None

##################################################

## Step 1: Read CSV File
df = pd.read_csv(r"movie_dataset.csv")
#print(df.head())
#.head() prints first few rows

## Step 2: Select Features
features = ['keywords', 'cast', 'genres', 'director']

## Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
    except Exception as e:
        print("Error:", e, "Row:", row)

df["combined_features"] = df.apply(combine_features, axis=1)
# This created a new column called combined_features which 
# has the combined string for each row. apply() applies the
# function to each row, since axis is set to 1

#print(df["combined_features"].head())

## Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
#fit learns the unique words for each doc in text. transform 
#finds out the word count of each word in each doc. toarray()
#converts this into an array where each row is a doc and each
#col is a word. value gives count of the word.

## Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)

col1,col2=st.columns([6,3])

with col1:
    st.title("Welcome to NextWatch!")
    x=st.text_input("Enter the full name of your favourite movie:")
with col2:
    st.image("D:\Screenshot (1).png", use_column_width=True)
    st.image("D:\pcf.jpg", use_column_width=True)

x=x.strip()


## Step 6: Get index of this movie from its title
if st.button('Recommend'):
    movie_index = get_index_from_title(x)

    if movie_index is None:
        st.write(f"Sorry, the movie '{x}' is not found in the database.")
    else:
        similar_movies = list(enumerate(cosine_sim[movie_index]))

        ## Step 7: Get a list of similar movies in descending order of similarity score
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

        ## Step 8: Print titles of first 50 movies
        st.write(f"30 Movies similar to '{x}':")
        i = 0
        for movie in sorted_similar_movies[1:]:
            st.write(get_title_from_index(movie[0]))
            i += 1
            if i >= 30:
                break





