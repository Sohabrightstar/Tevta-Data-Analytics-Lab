import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# TITANIC DATASET
# ==========================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

titanic = pd.read_csv(url)

print("FIRST 10 ROWS")
print(titanic.head(10))

print("\nDATA TYPES")
print(titanic.dtypes)

print("\nSHAPE")
print(titanic.shape)

numeric_cols = titanic.select_dtypes(include='number')

print("\nMEAN")
print(numeric_cols.mean())

print("\nMEDIAN")
print(numeric_cols.median())

print("\nSTANDARD DEVIATION")
print(numeric_cols.std())

print("\nMIN")
print(numeric_cols.min())

print("\nMAX")
print(numeric_cols.max())

print("\nSKEWNESS")
print(numeric_cols.skew())

print("\nKURTOSIS")
print(numeric_cols.kurt())

# Histogram

plt.figure(figsize=(8,5))
titanic['Age'].hist(bins=20, edgecolor='black')
plt.title("Titanic Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Survival by Passenger Class

plt.figure(figsize=(8,5))

survival = titanic.groupby('Pclass')['Survived'].sum()

survival.plot(kind='bar')

plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survivors")
plt.show()

# ==========================================
# SPOTIFY DATASET
# ==========================================

spotify = pd.read_csv("spotify_songs.csv")

audio_features = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]

corr_matrix = spotify[audio_features].corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Spotify Correlation Heatmap")
plt.show()

# ==========================================
# WORLD HAPPINESS
# ==========================================

happy = pd.read_csv("2023.csv")

top20 = happy.sort_values(
    by='Ladder score',
    ascending=False
).head(20)

plt.figure(figsize=(10,8))

sns.barplot(
    x='Ladder score',
    y='Country name',
    data=top20
)

plt.title("Top 20 Happiest Countries")
plt.show()
