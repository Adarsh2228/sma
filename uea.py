import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# === CONFIGURATION: Change column names as per your dataset ===
content_col = "Content"            # Text/Comment/Content column
likes_col = "Likes"
comments_col = "Comments"
shares_col = "Shares"
followers_col = "Followers"
content_type_col = "Content_Type"  # Optional (e.g., Video, Image, Text)

# === Load Dataset ===
df = pd.read_csv("your_dataset.csv")  # Replace with actual filename
df = df.dropna(subset=[content_col, likes_col, comments_col, shares_col, followers_col])

# === Clean + Basic EDA ===
df[content_col] = df[content_col].astype(str)
df[[likes_col, comments_col, shares_col, followers_col]] = df[[likes_col, comments_col, shares_col, followers_col]].fillna(0)

# === Compute Sentiment from Content ===
df["polarity"] = df[content_col].apply(lambda x: TextBlob(x).sentiment.polarity)
df["sentiment"] = df["polarity"].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

# === Compute Engagement Metric ===
df["engagement"] = (df[likes_col] + df[comments_col] + df[shares_col]) / df[followers_col].replace(0, 1)

# === VISUALIZATIONS (5 Graphs) ===

# 1. Sentiment Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="sentiment", palette="coolwarm")
plt.title("Sentiment Distribution of Content")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# 2. Engagement by Sentiment
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="sentiment", y="engagement", palette="Set2")
plt.title("User Engagement by Sentiment")
plt.ylabel("Engagement Rate")
plt.xlabel("Sentiment")
plt.show()

# 3. Average Engagement by Content Type (if available)
if content_type_col in df.columns:
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x=content_type_col, y="engagement", estimator='mean', palette="viridis")
    plt.title("Average Engagement by Content Type")
    plt.ylabel("Engagement Rate")
    plt.xlabel("Content Type")
    plt.xticks(rotation=45)
    plt.show()

# 4. Correlation Heatmap of Engagement Factors
plt.figure(figsize=(8,6))
sns.heatmap(df[[likes_col, comments_col, shares_col, followers_col, "engagement"]].corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Between Engagement Factors")
plt.show()

# 5. Engagement Distribution (Histogram)
plt.figure(figsize=(8,5))
sns.histplot(df["engagement"], bins=30, kde=True, color="orange")
plt.title("Distribution of Engagement Rate")
plt.xlabel("Engagement Rate")
plt.ylabel("Frequency")
plt.show()
