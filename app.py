from flask import Flask, render_template, request
import pandas as pd
import joblib
from PIL import Image
import os

app = Flask(__name__)

# Load the dataset and model
df = pd.read_csv('movies_recommendation1.csv')
count_vectorizer = joblib.load('count_vectorizer.pkl')
cosine_sim = joblib.load('cosine_similarity.pkl')

# Function to get movie recommendations
def get_recommendations(title):
    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:9]  # Get 8 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    if request.method == 'POST':
        movie_title = request.form.get('movie_title')
        recommendations = get_recommendations(movie_title)
    return render_template('index.html', recommendations=recommendations, movies=df['title'].tolist())
