#! "C:\Zahra\Uni_Verona\Programming\Project\Programming Project Spotify\Spotify-Project\myenv\Scripts\python.exe"
#_________________________________________________________________________________________________ 
# import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st

# streamlit run Streamlit-spotifydata.py
# myenv\Scripts\activate
#_________________________________________________________________________________________________ 
from streamlit_option_menu import option_menu
with st.sidebar:
    selected = option_menu(
        menu_title = 'Menu',
        options = ['Home', 'Data Exploration', 'Data Analysis', 'Machine Learning'],
        icons = ['house-door', 'table', 'pie-chart-fill', 'graph-up'],
        default_index = 0
    )

if selected == 'Home':
    st.markdown("<h1 style='text-align: center;'>Spotify Dataset 2023</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<h2 style='text-align: center;'>Programming for Data Science Final Project 2024</h2>", unsafe_allow_html=True)
    st.write("")
    st.write('<p style="text-align: center; font-size: 20px;">Zahra Nahardani</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 18px;"><a href="mailto:zahra.nahardani@studenti.univr.it">zahra.nahardani@studenti.univr.it</a></p>', unsafe_allow_html=True)
    
#__________________________________________________________________________________________________________
    
# Read the Spotify dataset from a CSV file into a Pandas DataFrame.
spotify_data_df = pd.read_csv('SpotifyData2023.csv', encoding='ISO-8859-1', low_memory=False)
spotify_data_df_copy = spotify_data_df.copy()

if selected == 'Data Exploration':
    st.subheader('Display Dataset')

    # Print the shape of the DataFrame to get the number of rows and columns.
    st.write('<b>Shape of Dataset:</b>', spotify_data_df_copy.shape, unsafe_allow_html=True)

    # Display the first and last 5 rows of the DataFrame to get an overview of the data.
    option = st.selectbox('Select an option', ['First 5 rows of the dataset', 'Last 5 rows of the dataset'])
    if option == 'First 5 rows of the dataset':
        st.write(spotify_data_df_copy.head(5))
    elif option == 'Last 5 rows of the dataset':
        st.write(spotify_data_df_copy.tail(5))
    
    # Generate descriptive statistics of the numerical columns in the DataFrame.
    st.write('<b>Dataset describtion</b>', spotify_data_df_copy.describe(), unsafe_allow_html=True)

    # Display concise information about the DataFrame, including data types and memory usage.
    spotify_data_df_copy.info()

# The number of missing values for each feature
missing_values = spotify_data_df_copy.isnull().sum()

# Drop missing values for the main genre
spotify_data_df_copy.dropna(subset=['genre_0'], inplace=True)

# Imput missing values for the artist, other genre columns, analysis_url, track_href, uri and  with unknown
unknown_features = ['artist_0','artist_1','artist_2','artist_3','artist_4','genre_1','genre_2','genre_3','genre_4',
                    'analysis_url','track_href','uri']
for feature in unknown_features:
        spotify_data_df_copy[feature].fillna('Unknown',inplace=True)

# Impute missing values for continuous variables with mean
continuous_features = ['acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness',
                       'speechiness','tempo','valence']
for feature in continuous_features:
    spotify_data_df_copy[feature].fillna(spotify_data_df_copy[feature].mean(), inplace=True)

# Impute missing values for categorical and ordinal variables with mode
categorical_features = ['mode','key','time_signature']
for feature in categorical_features:
    spotify_data_df_copy[feature].fillna(spotify_data_df_copy[feature].mode()[0], inplace=True)

# Impute missing values for type with the only value "audio_features"
spotify_data_df_copy['type'].fillna('audio_features',inplace=True)

# Remove missing values (with number of less than 60)
features_with_small_number_of_missing = ['album_name','artist_0','label','release_date','track_name','explicit',
                                         'track_popularity','release_year','release_month']
spotify_data_df_copy.dropna(subset=features_with_small_number_of_missing, inplace=True)

# Convert 'TRUE' to 1 and 'FALSE' to 0 in the 'explicit' column
unique_values = spotify_data_df_copy['explicit'].unique()
print(unique_values)
spotify_data_df_copy['explicit'] = spotify_data_df_copy['explicit'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}) 

if selected == 'Data Exploration':

    # Cleaning Data - Null Values Handling
    st.subheader('Data Cleaning')

    # Number of missing values before cleaning data
    if st.button('Number of null values before cleaning'):
        missing_value_info = ", ".join([f"{column}: <span style='color: blue;'>{count}</span>" for column,
                                         count in missing_values.items()])
        st.write(missing_value_info, unsafe_allow_html=True)
     
    # Number of missing values after cleaning data
    missing_values_after_cleaning = spotify_data_df_copy.isnull().sum()
    if st.button('Number of null values after cleaning'):
        missing_value_info = ", ".join([f"{column}: <span style='color: blue;'>{count}</span>" for column,
                                         count in missing_values_after_cleaning.items()])
        st.write(missing_value_info, unsafe_allow_html=True)

# Generate descriptive statistics and display concise information about the new DataFrame
print(spotify_data_df_copy.shape)
spotify_data_df_copy.describe()
spotify_data_df_copy.info()

if selected == 'Data Exploration':

    # Filter based on track popularity
    st.write('<b>Filter by track popularity</b>', unsafe_allow_html=True)
    selected_track_popularity = st.number_input('Select a track popularity number between 0 and 100 (100 being the most popular)',
                                                 min_value=0, max_value=100, value=None, step=1)
    if selected_track_popularity is not None:
        selected_rows = spotify_data_df_copy[spotify_data_df_copy['track_popularity'] == selected_track_popularity]
        st.write(selected_rows)

# Filter based on track release year

if selected == 'Data Exploration':
    st.write('<b>Filter by release year</b>', unsafe_allow_html=True)
    release_year = st.number_input('Enter the year:', 
                                   min_value=int(spotify_data_df_copy['release_year'].min()),
                                   max_value=int(spotify_data_df_copy['release_year'].max()), 
                                   )
    release_year = int(release_year)
    filtered_data_by_year = spotify_data_df_copy[spotify_data_df_copy['release_year'] == release_year]
    most_popular_track = filtered_data_by_year.nlargest(1, 'track_popularity')[['track_name', 'artist_0', 'track_popularity']]
    st.write(f"Most popular track in {release_year}:")
    st.write(most_popular_track)
#__________________________________________________________________________________________________________ 
 
# Count the number of unique genres in the 'genre_0' column
num_genres = spotify_data_df_copy['genre_0'].nunique()
print(num_genres)

# Show the frequencies of each genre
genre_counts = spotify_data_df_copy['genre_0'].value_counts()
print(genre_counts)

if selected == 'Data Analysis': 

    # Bar chart for the top 10 genres
    st.subheader('Bar chart')
    st.write('The frequency distribution of the top 10 main genres:')
    top_10_genres_counts = spotify_data_df_copy['genre_0'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10_genres_counts.plot(kind='bar', ax=ax, color='orchid')
    plt.title('10 Top Main Genres')
    plt.xlabel('Main Genre')
    plt.ylabel('Frequency')
    plt.xticks(rotation=60)
    st.pyplot(fig)
    plt.show()

    # Pie Chart for the top 10 genres
    st.subheader('Pie Chart')
    fig, ax = plt.subplots(figsize=(8,8))
    explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    colors = ['lightpink','thistle','lavender','aliceblue','lightblue','lightskyblue','lightCyan','lightgreen','paleturquoise','turquoise']
    ax.pie(top_10_genres_counts, labels=top_10_genres_counts.index, explode=explode, autopct='%1.1f%%', colors=colors)
    ax.set_title('Top 10 Main Genres Distribution')
    st.pyplot(fig)
    plt.show()
    
    # Grouped bar chart comparing track popularity for the top 10 genres
    st.subheader('Grouped bar chart')
    st.write('Comparing track popularity for the top 10 genres:')
    top_10_genres = spotify_data_df_copy['genre_0'].value_counts().head(10).index
    top_10_data = spotify_data_df_copy[spotify_data_df_copy['genre_0'].isin(top_10_genres)]
    grouped_data = top_10_data.groupby('genre_0')['track_popularity'].mean().reset_index().sort_values('track_popularity', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(grouped_data['genre_0'], grouped_data['track_popularity'], color='turquoise')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Average Track Popularity')
    ax.set_title('Average Track Popularity for Top 10 Genres')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.show()

    # Distribution of Track Popularity
    st.subheader('Hisogram')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(spotify_data_df_copy['track_popularity'], kde=True, color='plum', bins=40, ax=ax)
    ax.set_title('Distribution of Track Popularity')
    ax.set_xlabel('Track Popularity')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    plt.show()

    # Relationship between followers and album, track, and artist popularity
    st.subheader('Subplot')
    st.write('Relationship between followers and album, track, and artist popularity')
    fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(8,6))
    ax_1.scatter(spotify_data_df_copy['track_popularity'], spotify_data_df_copy['followers'],
              label='Track Popularity Vs Followers', color='olive')
    ax_1.set_xlabel('Track Popularity')

    ax_2.scatter(spotify_data_df_copy['artist_popularity'], spotify_data_df_copy['followers'],
              label='Artist Popularity Vs Followers', color='seagreen')
    ax_2.set_xlabel('Artist Popularity')

    ax_3.scatter(spotify_data_df_copy['album_popularity'], spotify_data_df_copy['followers'],
              label='Album Popularity Vs Followers', color='darkkhaki')
    ax_3.set_xlabel('Album Popularity')

    fig.text(0.06, 0.5, 'Followers', va='center', rotation='vertical')
    st.pyplot(fig)
    plt.show()

    # Correlation matrix
    st.subheader('Correlation Matrix')
    selected_features = ['album_popularity','artist_popularity','track_popularity','followers','danceability','energy','acousticness',
                         'instrumentalness', 'liveness','loudness','speechiness','valence','tempo']
    correlation_matrix = spotify_data_df_copy[selected_features].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    ax.set_title('Correlation')
    st.pyplot(fig)
    plt.show()

#__________________________________________________________________________________________________________ 
# Machine Learning

####### Linear Regression #######
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Split the data into features (x) and target (y)
x_features = spotify_data_df_copy[['album_popularity','artist_popularity','followers','loudness','danceability','energy']]
y_feature = spotify_data_df_copy['track_popularity']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_features, y_feature, test_size=0.2, random_state=60)

# Create and train the Linear Regression model
model_linear_Reg = LinearRegression()
model_linear_Reg.fit(x_train, y_train)

# Make predictions
y_predict = model_linear_Reg.predict(x_test)

# Evaluate the Linear Regression model 
# MSE
mean_sqe = mean_squared_error(y_test, y_predict)
print(f'Mean Squared Error: {mean_sqe}')

# R-squared
r_squared = r2_score(y_test, y_predict)
print(f'R-squared (R2) Score: {r_squared}')

if selected == 'Machine Learning': 
    st.subheader('Linear Regression')
    st.write('<b>Mean Squared Error (MSE):</b>', mean_sqe, unsafe_allow_html=True)
    st.write('<b>R-squared (R2) Score:</b>', r_squared, unsafe_allow_html=True)
    st.write('This model explains approximately 75% of the variance in track popularity based on the selected features. The Mean Squared Error (MSE) is relatively high at 80.51, indicating that the models predictions have a significant amount of error.')

####### Random Forest Regression #######
from sklearn.ensemble import RandomForestRegressor

model_random_forest = RandomForestRegressor(random_state=1)
model_random_forest.fit(x_train, y_train)

y_predict_rf = model_random_forest.predict(x_test)

# Evaluate the Random Forest model
# MSE
mean_sqe_rf = mean_squared_error(y_test, y_predict_rf)
print(f'Mean Squared Error (Random Forest): {mean_sqe_rf}')
 
# R-squared
r_squared_rf = r2_score(y_test, y_predict_rf)
print(f'R-squared (R2) Score: {r_squared_rf}')

if selected == 'Machine Learning': 
    st.subheader('Random Forest Regression')
    st.write('<b>Mean Squared Error (MSE):</b>', mean_sqe_rf, unsafe_allow_html=True)
    st.write('<b>R-squared (R2) Score:</b>', r_squared_rf, unsafe_allow_html=True)
    st.write("With an MSE of 59.10, the model's predictions have less error compared to the linear regression model. Also," 
             "An R-squared score of 0.8173 means that approximately 82% of the variance in track popularity is explained by"
             "the features used in the Random Forest model.")


    # Scatter plot of actual vs. predicted values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_predict_rf, color='Cyan', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='red', linewidth=2)  # Diagonal line
    ax.set_xlabel('Actual Track Popularity')
    ax.set_ylabel('Predicted Track Popularity')
    ax.set_title('Random Forest Regression: Predicted vs. Actual Values')
    ax.grid(True)
    st.pyplot(fig)
    plt.show()








    



    







