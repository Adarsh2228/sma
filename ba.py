import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from collections import Counter

# Download necessary NLTK data (first-time only)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Replace with actual filename

# 🔧 Modify these according to your dataset column names
TEXT_COL = 'Text'
SENTIMENT_COL = 'Sentiment'
DATE_COL = 'Timestamp'
HASHTAG_COL = 'Hashtags'

# Step 1: Clean & Pre-process
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def pre_process(text):
    if pd.isnull(text): return []
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in stop_words]

df['tokens'] = df[TEXT_COL].apply(pre_process)

# Fill NaNs if any
df.fillna('', inplace=True)

# Optional: Add new sentiment score from TextBlob
df['Sentiment_Polarity'] = df[TEXT_COL].apply(lambda x: TextBlob(x).sentiment.polarity)
df['VADER_Compound'] = df[TEXT_COL].apply(lambda x: sia.polarity_scores(x)['compound'])

# Convert timestamp to datetime
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')

# -------------------------------
# 📊 Exploratory Graphs (5+ Total)
# -------------------------------

# 1️⃣ WordCloud
all_text = ' '.join(df[TEXT_COL])
wordcloud = WordCloud(width=800, height=500, background_color='white').generate(all_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Word Cloud of Brand Mentions")
plt.axis('off')
plt.show()

# 2️⃣ Bar Plot of Sentiment Counts
sentiment_counts = df[SENTIMENT_COL].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
plt.figure(figsize=(8, 6))
sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, palette='Set2')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# 3️⃣ Histogram of TextBlob Sentiment Polarity
plt.figure(figsize=(10, 6))
sns.histplot(df['Sentiment_Polarity'], bins=30, kde=True, color='orange')
plt.title("Distribution of Sentiment Polarity (TextBlob)")
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.show()

# 4️⃣ Time-based Analysis (Posts per Day/Hour)
if df[DATE_COL].dtype == 'datetime64[ns]':
    df['Date'] = df[DATE_COL].dt.date
    df['Hour'] = df[DATE_COL].dt.hour
    time_series = df.groupby('Date').size()
    plt.figure(figsize=(12, 6))
    time_series.plot()
    plt.title("Posts Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Posts")
    plt.grid()
    plt.show()

# 5️⃣ Top Hashtags
hashtags = []
for text in df[HASHTAG_COL].dropna():
    hashtags.extend(re.findall(r'#\w+', str(text)))
hashtag_freq = Counter(hashtags).most_common(10)
tags, counts = zip(*hashtag_freq)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(tags), y=list(counts), palette='magma')
plt.title("Top 10 Hashtags")
plt.xlabel("Hashtag")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6️⃣ Most Common Words (from tokens)
all_tokens = [token for sublist in df['tokens'] for token in sublist]
common_words = Counter(all_tokens).most_common(10)
words, freqs = zip(*common_words)
plt.figure(figsize=(8, 5))
sns.barplot(x=list(words), y=list(freqs), palette='coolwarm')
plt.title("Most Common Words")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.show()

# 7️⃣ Heatmap of Numerical Correlation
numerical_cols = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numerical Features")
plt.show()

# Save cleaned version (optional)
df.to_csv("cleaned_brand_data.csv", index=False)

# ✔ Summary Done!
print("EDA and Brand Analysis Complete. Modify only column names to reuse.")
