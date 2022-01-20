# Importing the libraries we will use
import pandas as pd
import numpy as np
import pickle
import re
import mysql.connector
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Connecting to the database
# Enter your own host name, user name, password, database name
mymoviesafrica = mysql.connector.connect(host="localhost", user="root", passwd="#####", database="mymoviesafrica")

# Uploading the content table
content = pd.read_sql("SELECT id, title, synopsis, genres, tags FROM content", mymoviesafrica)

# Data cleaning
# Removing actor's names from the synopsis which are in brackets. Their names are already in tags
content['synopsis'] = content['synopsis'].apply(lambda x : re.sub(r"\([^()]*\)", "", x))

# Lowercasing and removing all punctuation marks
content['synopsis'] = content['synopsis'].apply(lambda x : str.lower(re.sub('[^\w\s]', '', x)))

# Removing stop words such as the, a, an
stop = stopwords.words('english')
content['synopsis'] = content['synopsis'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Cleaning genre
# Removing the quotation marks
content['genres'] = content['genres'].apply(lambda x: x.replace('"', ''))

# Convert genres into a list of strings
content['genres'] = content['genres'].apply(lambda x:x[1:-1].split(','))

# Filtering out the main genres
genres = ["Action", "Drama", "Romance", "Comedy", "Crime", "Family", "Adventure", "Thriller", "Suspense", "Supernatural",
"Political", "Activism", "True Story", "Chick Flick", "Sports", "Short & Sweet", "Feel Good", "Musical", "Animation", 
"History", "Super Hero", "Fantasy", "Feel Good", ]
content['genres'] = content['genres'].apply(lambda x :[i for i in x if i in genres])

# lowercasing the genres
content['genres'] = content['genres'].apply(lambda x : [str.lower(i.replace(" ", "")) for i in x])

# Converting the list to strings for easier joining later on
content['genres'] = [' '.join(map(str, l)) for l in content['genres']]

# Cleaning tags
# Removing the spacing between words 
# so that the vectorizer does not count the Brenda in "Brenda Wairimu" and "Brenda Shiru" as the same
content['tags'] = content['tags'].apply(lambda x : str.lower(x.replace(" ", "").replace(",", " ")))

# Joining all the columns together by a space so that we can use all the columns in making recommendations
# Below is a function that we do this
def create_soup(x):
    return ''.join(x['synopsis']) + ' ' + ''.join(x['genres']) + ' ' + ''.join(x['tags'])

# New column with the soup
content['soup'] = content.apply(create_soup, axis=1)

# Using synopsis, genres and tags to make general recommendations
soup_count = CountVectorizer()
soup_matrix = soup_count.fit_transform(content['soup'])
soup_similarity = pd.DataFrame(cosine_similarity(soup_matrix, soup_matrix), index=content['id'].values, columns=content['id'].values)

# Pickling the similarity dataframe for use in deployment.
soup_similarity.to_pickle("./soup_similarity.pkl")

# Function that outputs the five most similar movies in a list
def general_recommender(title):
    title_id = content[content['title'] == title]['id'].values
    similar_id = soup_similarity[title_id].nlargest(6, title_id).index[1:]
    similar_title = content[content['id'].isin(similar_id)]['title'].values.tolist()
    return similar_title

# Using synopsis only to make recommendations
tfidf = TfidfVectorizer()
#Replace NaN with an empty string
content['synopsis'] = content['synopsis'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(content['synopsis'])

# Compute the cosine similarity matrix
synopsis_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

# Converting it to a dataframe with the movie ids as the indices and columns
synopsis_similarity = pd.DataFrame(synopsis_similarity, columns=content['id'].values, index=content['id'].values)

# Pickling the similarity dataframe for use in deployment.
synopsis_similarity.to_pickle("./synopsis_similarity.pkl")

# Function that outputs the five most similar movies in a list
def synopsis_recommender(title):
    title_id = content[content['title'] == title]['id'].values
    similar_id = synopsis_similarity[title_id].nlargest(6, title_id).index[1:]
    similar_title = content[content['id'].isin(similar_id)]['title'].values.tolist()
    return similar_title

# Using genre only to make recommendations
genre_count = CountVectorizer()
genre_matrix = genre_count.fit_transform(content['genres'])
genre_similarity = pd.DataFrame(cosine_similarity(genre_matrix, genre_matrix), index=content['id'].values, columns=content['id'].values)

# Pickling the similarity dataframe for use in deployment.
genre_similarity.to_pickle("./genre_similarity.pkl")

# Function
def genre_reccomender(title):
    title_id = content[content['title'] == title]['id'].values
    similar_id = genre_similarity[title_id].nlargest(6, title_id).index[1:]
    similar_title = content[content['id'].isin(similar_id)]['title'].values.tolist()
    return similar_title

# Using tags i.e. to make recommendations
tag_count = CountVectorizer()
tag_matrix = tag_count.fit_transform(content['tags'])
tag_similarity = pd.DataFrame(cosine_similarity(tag_matrix, tag_matrix), index=content['id'].values, columns=content['id'].values)

# Pickling the similarity dataframe for use in deployment.
tag_similarity.to_pickle("./tag_similarity.pkl")

# Function
def tag_reccomender(title):
    title_id = content[content['title'] == title]['id'].values
    similar_id = tag_similarity[title_id].nlargest(6, title_id).index[1:]
    similar_title = content[content['id'].isin(similar_id)]['title'].values.tolist()
    return similar_title

# Once you have the directors data in a more suitable form you can use this function to output movies with similar directors
# Make sure the column name for directors matches the one below
def director_recommender(title):
    director = content[content['title'] == title]['director'].values.tolist()
    director_movies = content[content['director'].isin(director)][['title']]
    return director_movies[director_movies['title'] != title]['title'].values.tolist()

# The same applies for recommendation based on production company
def prod_company_recommender(title):
    director = content[content['title'] == title]['production_company'].values.tolist()
    director_movies = content[content['production_company'].isin(director)][['title']]
    return director_movies[director_movies['title'] != title]['title'].values.tolist()



