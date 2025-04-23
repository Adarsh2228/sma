import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Dataset
df = pd.read_csv("social_media_data.csv")  # Change this to your actual file
text_col = df.columns[0]  # assuming first column is text
df = df[[text_col]].dropna().rename(columns={text_col: "text"})

# Clean Text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^A-Za-z\s]", '', text)
    text = text.lower()
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ----------- EDA Visualizations -------------
# 1. Word count distribution
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8,5))
sns.histplot(df['word_count'], bins=20, kde=True)
plt.title("Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# 2. Top 20 frequent words
from collections import Counter
all_words = " ".join(df['clean_text']).split()
top_words = Counter(all_words).most_common(20)
words, counts = zip(*top_words)

plt.figure(figsize=(10,5))
sns.barplot(x=list(counts), y=list(words))
plt.title("Top 20 Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.show()

# 3. WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_words))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud")
plt.show()

# ----------- Keyword Extraction (TF-IDF) -------------
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])

# 4. Top TF-IDF Words
feature_names = tfidf_vectorizer.get_feature_names_out()
mean_tfidf = X_tfidf.mean(axis=0).A1
top_tfidf_words = sorted(zip(mean_tfidf, feature_names), reverse=True)[:20]
scores, keywords = zip(*top_tfidf_words)

plt.figure(figsize=(10,5))
sns.barplot(x=list(scores), y=list(keywords))
plt.title("Top 20 TF-IDF Keywords")
plt.xlabel("Average TF-IDF Score")
plt.ylabel("Keywords")
plt.show()

# ----------- Topic Modeling (LDA) -------------
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X_tfidf)

# 5. Topics with Top Words
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

print("\nLDA Topics:")
display_topics(lda_model, feature_names, 10)

# Optional: PyLDAvis (for notebook/HTML)
# import pyLDAvis.sklearn
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.sklearn.prepare(lda_model, X_tfidf, tfidf_vectorizer)
# pyLDAvis.display(vis)
