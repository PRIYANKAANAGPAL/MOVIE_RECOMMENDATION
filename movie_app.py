import streamlit as st # type: ignore
import pandas as pd # type: ignore
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data
movies_data = pd.read_csv("movies.csv")

# Combine selected features into one column
selected_features = ["genres", "keywords", "tagline", "cast", "director"]
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna(" ")
combined_features = movies_data["genres"] + " " + movies_data["keywords"] + " " + \
                    movies_data["tagline"] + " " + movies_data["cast"] + " " + \
                    movies_data["director"]

# Convert text data to vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

# Streamlit app
def main():
    st.title("Movie Recommendation System")

    # User input
    movie_name = st.text_input("Enter your favorite movie:")
    
    if st.button("Get Recommendations"):
        # Find close matches for the entered movie name
        close_matches = difflib.get_close_matches(movie_name, movies_data["title"].tolist())
        
        if close_matches:
            first_match = close_matches[0]
            index_of_the_movie = movies_data[movies_data.title == first_match].index[0]
            similarity_scores = list(enumerate(similarity[index_of_the_movie]))
            sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Display movie recommendations
            st.subheader("Recommended Movies:")
            for i, (index, score) in enumerate(sorted_scores[:10], start=1):
                title = movies_data.loc[index, "title"]
                st.write(f"{i}. {title}")
        else:
            st.write("No close matches found for the entered movie name.")

if __name__ == "__main__":
    main()
