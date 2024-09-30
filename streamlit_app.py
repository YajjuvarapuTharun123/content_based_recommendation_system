import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load the dataset
df = pd.read_csv('movies_recommendation.csv')

# Load the model
count_vectorizer = joblib.load('count_vectorizer.pkl')
cosine_sim = joblib.load('cosine_similarity.pkl')

# Function to get movie recommendations
def get_recommendations(title):
    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:9]  # Get 5 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

st.title('Telugu Movie Recommendation System')

# User input for movie title
movie_title = st.selectbox('Select a movie:', df['title'])

if st.button('Recommend'):
    recommendations = get_recommendations(movie_title)
    st.subheader('Recommended Movies:')
    
    # Create columns for displaying images
    cols = st.columns(4)
    
    for i, movie in enumerate(recommendations.itertuples()):
        with cols[i % 4]:  # Distribute movies across columns
            st.write(movie.title)
            img_path = movie.image
            image = Image.open(img_path)

            # Resize the image
            image = image.resize((150, 200))  # Set width=150 and height=200
            st.image(image, caption=movie.title, use_column_width=False)

