
# Movie Recommendation System

A **content-based movie recommendation system** that suggests movies similar to a user's favorite movie using cosine similarity on features like director, tags, genre, and description.

## Features

* Recommends movies based on content similarity rather than user ratings.
* Uses **cosine similarity** to compute similarity between movies.
* Considers multiple features: **director, tags, genre, and description**.
* Returns movies sorted by descending similarity score.
* Interactive **Streamlit UI** for easy input and displaying recommendations.

## How It Works

1. The system creates a combined textual profile of each movie from its director, genre, tags, and description.
2. It vectorizes these profiles using TF-IDF or a similar text vectorization method.
3. Cosine similarity is computed between the vector of the user's favorite movie and all other movies.
4. The top-N most similar movies are returned as recommendations.

