import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# 🔧 Change these column names as per your dataset
TEXT_COL = 'text'
DATE_COL = 'date'
COMPETITOR_COL = 'competitor'
LIKES_COL = 'likes'
COMMENTS_COL = 'comments'
SHARES_COL = 'shares'
SENTIMENT_COL = 'sentiment'
PLATFORM_COL = 'platform'

# Load dataset
df = pd.read_csv("your_dataset.csv")  # <- Change filename here
df.fillna('', inplace=True)

# Convert date to datetime
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')

# Preprocessing for keyword extraction
stop_words = set(stopwords.words('english'))

def extract_keywords(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]

df['keywords'] = df[TEXT_COL].apply(extract_keywords)

# -----------------------------
# 📊 5+ Graphs for Competitor Analysis
# -----------------------------

# 1️⃣ Avg Likes, Comments, Shares per Competitor
agg_metrics = df.groupby(COMPETITOR_COL)[[LIKES_COL, COMMENTS_COL, SHARES_COL]].mean()
print("Average Engagement Metrics per Competitor:")
print(agg_metrics)

agg_metrics.plot(kind='bar', figsize=(10,6), title="Avg Engagement per Competitor")
plt.ylabel("Average Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2️⃣ Sentiment distribution by Competitor
plt.figure(figsize=(10,6))
sns.countplot(data=df, x=COMPETITOR_COL, hue=SENTIMENT_COL)
plt.title("Sentiment Distribution by Competitor")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45)
plt.show()

# 3️⃣ Platform distribution by Competitor
plt.figure(figsize=(10,6))
sns.countplot(data=df, x=COMPETITOR_COL, hue=PLATFORM_COL)
plt.title("Platform Usage by Competitor")
plt.ylabel("Post Count")
plt.xticks(rotation=45)
plt.show()

# 4️⃣ Boxplot of Shares by Competitor
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x=COMPETITOR_COL, y=SHARES_COL)
plt.title("Shares Distribution by Competitor")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5️⃣ Time Series of Posts per Competitor
if DATE_COL in df:
    df['Day'] = df[DATE_COL].dt.date
    time_series = df.groupby(['Day', COMPETITOR_COL]).size().unstack().fillna(0)
    time_series.plot(figsize=(12,6))
    plt.title("Daily Post Activity by Competitor")
    plt.xlabel("Date")
    plt.ylabel("Number of Posts")
    plt.grid()
    plt.show()

# 6️⃣ Most Frequent Keywords Overall
all_keywords = [word for keywords in df['keywords'] for word in keywords]
top_keywords = Counter(all_keywords).most_common(15)
words, freqs = zip(*top_keywords)
plt.figure(figsize=(12,6))
sns.barplot(x=list(words), y=list(freqs), palette="Set3")
plt.title("Top 15 Keywords in Posts")
plt.xlabel("Keyword")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# ✅ Save cleaned version
df.to_csv("cleaned_sma_competitor_analysis.csv", index=False)
print("✅ Analysis completed. Ready for practical.")
