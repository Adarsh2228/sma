import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

# === CONFIGURATION: Change these column names as per your dataset ===
user_group_col = "Gender"      # E.g. 'Gender', 'Location', 'Age_Group'
hashtag_col = "Hashtags"       # If hashtags are in a separate column
text_col = "Comments"          # If hashtags are inside text/comments

# === Load Dataset ===
df = pd.read_csv("your_dataset.csv")  # Replace with your file
df = df[[user_group_col, hashtag_col, text_col]].dropna()

# === Extract hashtags from either 'Hashtags' column or 'Comments' ===

# Helper: Extract hashtags from text
def extract_hashtags(text):
    return re.findall(r"#\w+", str(text).lower())

# Case 1: Hashtags column exists (space or # separated)
df["all_hashtags"] = df[hashtag_col].fillna("").apply(lambda x: extract_hashtags(x))

# Case 2: Extract from text/comment if no hashtags present
if df["all_hashtags"].apply(len).sum() == 0:
    df["all_hashtags"] = df[text_col].apply(lambda x: extract_hashtags(x))

# Flatten all hashtags
all_hashtags = [tag for sublist in df["all_hashtags"] for tag in sublist]

# === EDA + Visualizations ===

# 1. Top 15 hashtags overall
hashtag_freq = Counter(all_hashtags)
top_15 = dict(hashtag_freq.most_common(15))
plt.figure(figsize=(10, 6))
sns.barplot(x=list(top_15.values()), y=list(top_15.keys()), palette="viridis")
plt.title("Top 15 Hashtags Overall")
plt.xlabel("Frequency")
plt.ylabel("Hashtags")
plt.show()

# 2. Wordcloud of hashtags
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(hashtag_freq)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Hashtag WordCloud")
plt.show()

# 3. Popular hashtags by user group
group_hashtags = df.explode("all_hashtags").groupby([user_group_col, "all_hashtags"]).size().reset_index(name="count")
top_grouped = group_hashtags[group_hashtags["count"] > 1]

plt.figure(figsize=(12, 6))
sns.barplot(data=top_grouped, x="all_hashtags", y="count", hue=user_group_col)
plt.title("Popular Hashtags by User Group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Most active user group (who posts most hashtags)
group_counts = df.groupby(user_group_col)["all_hashtags"].apply(lambda x: sum(len(i) for i in x)).reset_index(name="Total Hashtags")
plt.figure(figsize=(8, 5))
sns.barplot(data=group_counts, x=user_group_col, y="Total Hashtags", palette="coolwarm")
plt.title("Total Hashtags Used by User Groups")
plt.ylabel("Hashtag Count")
plt.show()

# 5. Heatmap of hashtag distribution across groups
pivot_table = group_hashtags.pivot_table(index="all_hashtags", columns=user_group_col, values="count", fill_value=0)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap="YlGnBu", annot=False)
plt.title("Hashtag Distribution Across User Groups")
plt.xlabel("User Group")
plt.ylabel("Hashtag")
plt.show()
