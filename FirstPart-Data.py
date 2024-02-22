#! "C:\Zahra\Uni_Verona\Programming\Project\Programming Project Spotify\Spotify-Project\myenv\Scripts\python.exe"

#_________________________________________________________________________________________________ 
# import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
#_________________________________________________________________________________________________ 
# Read the Spotify dataset from a CSV file into a Pandas DataFrame.
spotify_data_df = pd.read_csv('SpotifyData2023.csv', encoding='ISO-8859-1', low_memory=False)   

# Print the shape of the DataFrame to get the number of rows and columns.
print(spotify_data_df.shape)

# Display the first and last 5 rows of the DataFrame to get an overview of the data.
spotify_data_df.head(5)
spotify_data_df.tail(5)

# Generate descriptive statistics of the numerical columns in the DataFrame.
spotify_data_df.describe()

# Display concise information about the DataFrame, including data types and memory usage.
spotify_data_df.info()

#_________________________________________________________________________________________________ 
# Cleaning Data - Null Values Handling

spotify_data_df_copy = spotify_data_df.copy()

# The number of missing values for each feature
missing_values = spotify_data_df_copy.isnull().sum()
print(missing_values)

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

# Number of missing values after cleaning data
missing_values_after_cleaning = spotify_data_df_copy.isnull().sum()
print(missing_values_after_cleaning)

# Convert 'TRUE' to 1 and 'FALSE' to 0 in the 'explicit' column
unique_values = spotify_data_df_copy['explicit'].unique()
print(unique_values)

spotify_data_df_copy['explicit'] = spotify_data_df_copy['explicit'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})

#_________________________________________________________________________________________________
# Generate descriptive statistics and display concise information about the new DataFrame
print(spotify_data_df_copy.shape)
spotify_data_df_copy.describe().T
spotify_data_df_copy.info()

#_________________________________________________________________________________________________ 
# Visualization for the genre

# Count the number of unique genres in the 'genre_0' column
num_genres = spotify_data_df_copy['genre_0'].nunique()
print(num_genres)

# Show and plot the frequencies of each genre
genre_counts = spotify_data_df_copy['genre_0'].value_counts()
print(genre_counts)

# Bar Plot for the top 10 genres
top_10_genres_counts = spotify_data_df_copy['genre_0'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_10_genres_counts.plot(kind='bar')
plt.title('10 Top Main Genres')
plt.xlabel('Main Genre')
plt.ylabel('Frequency')
plt.xticks(rotation=60)
plt.show()

# Pie Chart for the top 10 genres
plt.figure(figsize=(10,6))
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
colors = ['lightpink','thistle','lavender','aliceblue','lightblue','lightskyblue','lightCyan','lightgreen','paleturquoise','turquoise']
plt.pie(top_10_genres_counts, labels=top_10_genres_counts.index, explode=explode, autopct='%1.1f%%', colors=colors)
plt.show()

# Grouped bar chart comparing track popularity for the top 10 genres
top_10_genres = spotify_data_df_copy['genre_0'].value_counts().head(10).index
top_10_data = spotify_data_df_copy[spotify_data_df_copy['genre_0'].isin(top_10_genres)]
grouped_data = top_10_data.groupby('genre_0')['track_popularity'].mean().reset_index().sort_values('track_popularity', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(grouped_data['genre_0'], grouped_data['track_popularity'], color='turquoise')
plt.xlabel('Genre')
plt.ylabel('Average Track Popularity')
plt.title('Average Track Popularity for Top 10 Genres')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#_________________________________________________________________________________________________ 
# Visualization for the popularity

# Relationship between followers and album, track, and artist popularity
fig_1 = plt.figure(figsize=(8,6))
ax_1 = fig_1.add_subplot(1, 3, 1)
ax_2 = fig_1.add_subplot(1, 3, 2)
ax_3 = fig_1.add_subplot(1, 3, 3)
ax_1.scatter(spotify_data_df_copy['track_popularity'], spotify_data_df_copy['followers'],
              label='Track Popularity Vs Followers', color='olive')
ax_1.set_xlabel('Track Popularity')

ax_2.scatter(spotify_data_df_copy['artist_popularity'], spotify_data_df_copy['followers'],
              label='Artist Popularity Vs Followers', color='seagreen')
ax_2.set_xlabel('Artist Popularity')

ax_3.scatter(spotify_data_df_copy['album_popularity'], spotify_data_df_copy['followers'],
              label='Album Popularity Vs Followers', color='darkkhaki')
ax_3.set_xlabel('Album Popularity')

fig_1.text(0.06, 0.5, 'Followers', va='center', rotation='vertical')
plt.show()

# Distribution of Track Popularity
plt.figure(figsize=(10, 6))
sns.histplot(spotify_data_df_copy['track_popularity'], kde=True, color='plum', bins=40)
plt.title('Distribution of Track Popularity')
plt.xlabel('Track Popularity')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
selected_features = ['album_popularity','artist_popularity','track_popularity','followers','danceability','energy','acousticness',
                     'instrumentalness', 'liveness','loudness','speechiness','valence','tempo']
correlation_matrix = spotify_data_df_copy[selected_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation')
plt.show()

#_________________________________________________________________________________________________ 
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

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_predict_rf, color='Cyan', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='red', linewidth=2)  # Diagonal line
plt.xlabel('Actual Track Popularity')
plt.ylabel('Predicted Track Popularity')
plt.title('Random Forest Regression: Predicted vs. Actual Values')
plt.grid(True)
plt.show()

