eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from scipy import stats
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Display the first few rows to understand the data structure
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handle missing values if needed (example: filling with 'unknown')
df.fillna('unknown', inplace=True)

# Basic descriptive statistics
print(df.describe())

# EDA: Mean, Median, Mode (for numerical columns like sentiments, etc.)
numerical_columns = ['sentiments']  # Change as per your dataset

for column in numerical_columns:
    print(f"Mean of {column}: {df[column].mean()}")
    print(f"Median of {column}: {df[column].median()}")
    print(f"Mode of {column}: {df[column].mode()[0]}")

# Distribution of numerical data (sentiments, or any other numerical column)
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiments'], kde=True)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap between numerical columns
corr_matrix = df[numerical_columns].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Word Cloud: For 'comments' or text-related columns
text_column = 'comments'  # Change to your actual column name for comments or text data
all_comments = ' '.join(df[text_column].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Comments')
plt.show()

# Sentiment Analysis: Count of positive, negative, and neutral sentiments
sentiment_counts = df['sentiments'].value_counts()  # Modify 'sentiments' based on your data column
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# Plot for Date (change column name to your date column if available)
df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' column is in datetime format
df.set_index('date', inplace=True)
df.resample('M').size().plot(figsize=(10, 6))
plt.title('Monthly Post Count')
plt.xlabel('Date')
plt.ylabel('Count')
plt.show()

# Additional Analysis: Top Keywords (based on text column like 'comments')
stop_words = set(stopwords.words('english'))
vectorizer = CountVectorizer(stop_words=stop_words, max_features=20)  # Top 20 words
X = vectorizer.fit_transform(df['comments'].dropna())
top_words = vectorizer.get_feature_names_out()

# Display Top 20 Keywords
top_keywords = np.array(X.sum(axis=0)).flatten()
top_words_df = pd.DataFrame({'Word': top_words, 'Frequency': top_keywords})
top_words_df = top_words_df.sort_values(by='Frequency', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Word', data=top_words_df)
plt.title('Top 20 Keywords in Comments')
plt.xlabel('Frequency')
plt.ylabel('Keyword')
plt.show()

# Other statistical insights like skewness or kurtosis
for column in numerical_columns:
    print(f"Skewness of {column}: {df[column].skew()}")
    print(f"Kurtosis of {column}: {df[column].kurt()}")

# Save cleaned dataset (optional)
df.to_csv('cleaned_data.csv', index=False)
