import pickle
import streamlit as st
import pandas as pd
import numpy as np
import random
from PIL import Image
# import mysql.connector
# from mysql.connector import Error

# # You can use the code below to connect directly to the database once you have the directors, production studio updated
# mymoviesafrica = mysql.connector.connect(host="localhost", user="root", passwd="langtouma", database="mymoviesafrica")
# content = pd.read_sql("SELECT id, title, synopsis, genres, tags FROM content", mymoviesafrica)

# But we will use a csv file with the dummy directors and production studios for this demonstration
content = pd.read_csv('content_with_dummies.csv')

# Importing the soup_similarity table we pickled for general recommendations
soup_similarity = pd.read_pickle("./soup_similarity.pkl")

# Importing the synopsis similarity table that we pickled for recommendations based on plot
synopsis_similarity = pd.read_pickle("./synopsis_similarity.pkl")

# Importing the genre similarity table that we pickled for recommendations based on genre
genre_similarity = pd.read_pickle("./genre_similarity.pkl")

# Importing the tag similarity table that we pickled for recommendations based on cast and director
tag_similarity = pd.read_pickle("./tag_similarity.pkl")


# Function for getting general recommendations
@st.cache
def general_recommender(title):
    title_id = content[content['title'] == title]['id'].values
    similar_id = soup_similarity[title_id].nlargest(6, title_id).index[1:]
    similar_title = content[content['id'].isin(similar_id)]['title'].values.tolist()
    return similar_title

# Function for getting recommendations based on plot
@st.cache
def synopsis_recommender(title):
    title_id = content[content['title'] == title]['id'].values
    similar_id = synopsis_similarity[title_id].nlargest(6, title_id).index[1:]
    similar_title = content[content['id'].isin(similar_id)]['title'].values.tolist()
    return similar_title

# Function for getting recommendations based on genre
@st.cache
def genre_recommender(title):
    title_id = content[content['title'] == title]['id'].values
    similar_id = genre_similarity[title_id].nlargest(6, title_id).index[1:]
    similar_title = content[content['id'].isin(similar_id)]['title'].values.tolist()
    return similar_title

# Function for getting recommendations based on cast and director
@st.cache
def tag_recommender(title):
    title_id = content[content['title'] == title]['id'].values
    similar_id = tag_similarity[title_id].nlargest(6, title_id).index[1:]
    similar_title = content[content['id'].isin(similar_id)]['title'].values.tolist()
    return similar_title

# Function for getting recommendations based on director
# Note: directors are dummy names
@st.cache
def director_recommender(title):
    director = content[content['title'] == title]['director'].values.tolist()
    director_movies = content[content['director'].isin(director)][['title']]
    return director_movies[director_movies['title'] != title]['title'].values.tolist()[0:5]

# Function for getting recommendations based on production house
# Note: production houses are dummy names
@st.cache
def prod_company_recommender(title):
    director = content[content['title'] == title]['production_company'].values.tolist()
    director_movies = content[content['production_company'].isin(director)][['title']]
    return director_movies[director_movies['title'] != title]['title'].values.tolist()[0:5]

# List of image directories
images = ['image1.jfif', 'image2.jfif', 'image3.jfif', 'image4.jpeg', 'image5.jfif']

# Title of the web app
st.title("MyMoviesAfrica Movie Recommender")

# Title of the sidebar
st.sidebar.header("Movie Search")

# Selecting a movie 
movie = st.sidebar.selectbox(label="Select Movie", options=content['title']) 

# Selecting the features to base the search on 
features = ['General', 'Plot', 'Genre', 'Cast', 'Director', 'Production Company']
selected_feature = st.sidebar.radio(label="Search based on which feature", options=features)

# Search button
search = st.sidebar.button('Search')

# Disclaimer 
st.sidebar.markdown("**_Note:_** the director and production house recommendations are based on dummy data")

if search:
    if selected_feature == 'General':
        st.subheader(f'Similar movies to "{movie}":')
        recommendations = general_recommender(movie)

    if selected_feature == 'Plot':
        st.subheader(f'Movies with a similar plot to "{movie}":')
        recommendations = synopsis_recommender(movie)
    
    if selected_feature == 'Genre':
        st.subheader(f'Similar genre as "{movie}":')
        recommendations = genre_recommender(movie)
    
    if selected_feature == 'Cast':
        st.subheader(f'Movies with a similar cast or director as "{movie}":')
        recommendations = tag_recommender(movie)
    
    if selected_feature == 'Director':
        director = content[content['title'] == movie]['director'].values[0]
        st.subheader(f'The director of "{movie}", {director}, has also directed:')
        recommendations = director_recommender(movie)

    if selected_feature == 'Production Company':
        company = content[content['title'] == movie]['production_company'].values[0]
        st.subheader(f'"{movie}" was produced by {company} who have also produced:')
        recommendations = prod_company_recommender(movie)

    col1, col2, col3, col4, col5 = st.columns(5)
    col_list = [col1, col2, col3, col4, col5]
    for i in range(len(recommendations)):
        with col_list[i]:
            st.image(Image.open(random.choice(images)), recommendations[i])

