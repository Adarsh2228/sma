import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms.community import girvan_newman
from sklearn.cluster import KMeans
import numpy as np

# 📌 Column Names – Modify Only These
SOURCE_COL = 'source'
TARGET_COL = 'target'
TEXT_COL = 'text'  # optional: if post/message columns exist
NUMERIC_COLS = ['weight']  # optional: numerical analysis if weights exist

# Load dataset
df = pd.read_csv("your_dataset.csv")  # 🔄 Update filename
print("\n📊 Dataset Preview:\n", df.head())

# ------------------ EDA ------------------
print("\n📌 Basic Info:")
print(df.info())
print("\n📊 Missing Values:\n", df.isnull().sum())

# Drop NA in source/target to avoid graph errors
df = df.dropna(subset=[SOURCE_COL, TARGET_COL])

# ----------------------------------------
# 🧠 Build Graph
# ----------------------------------------
G = nx.Graph()
edges = list(zip(df[SOURCE_COL], df[TARGET_COL]))
G.add_edges_from(edges)

print(f"\n🔗 Total Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# ------------------ Graph 1 ------------------
plt.figure(figsize=(10, 6))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=600)
plt.title("Original Social Network Graph")
plt.show()

# ----------------------------------------
# 👥 Community Detection using Girvan-Newman
# ----------------------------------------
communities = girvan_newman(G)
top_level_communities = next(communities)
community_list = [list(c) for c in top_level_communities]

print("\n🔍 Detected Communities:")
for i, com in enumerate(community_list):
    print(f"Community {i+1}: {com}")

# ------------------ Graph 2 ------------------
color_map = {}
colors = sns.color_palette("hsv", len(community_list))
for i, community in enumerate(community_list):
    for node in community:
        color_map[node] = colors[i]

node_colors = [color_map.get(node, "gray") for node in G.nodes()]
plt.figure(figsize=(10, 6))
nx.draw(G, with_labels=True, node_color=node_colors, node_size=600)
plt.title("Community Detection using Girvan-Newman")
plt.show()

# ----------------------------------------
# 🔢 Centrality Analysis
# ----------------------------------------
deg_cent = nx.degree_centrality(G)
btw_cent = nx.betweenness_centrality(G)

centrality_df = pd.DataFrame({
    "Node": list(deg_cent.keys()),
    "Degree Centrality": list(deg_cent.values()),
    "Betweenness Centrality": list(btw_cent.values())
}).sort_values(by="Betweenness Centrality", ascending=False)

print("\n🏆 Top Influential Nodes (Betweenness Centrality):")
print(centrality_df.head())

# ------------------ Graph 3 ------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=centrality_df.head(10), x="Betweenness Centrality", y="Node", palette="viridis")
plt.title("Top 10 Influential Nodes")
plt.show()

# ------------------ KMeans Clustering ------------------
features = centrality_df[["Degree Centrality", "Betweenness Centrality"]].values
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(features)

# ------------------ Graph 4 ------------------
plt.figure(figsize=(10, 6))
plt.scatter(features[:, 0], features[:, 1], c=labels, s=100, cmap="Set1")
for i, node in enumerate(centrality_df["Node"]):
    plt.text(features[i, 0], features[i, 1], node, fontsize=9, ha='right')
plt.xlabel("Degree Centrality")
plt.ylabel("Betweenness Centrality")
plt.title("KMeans Clustering of Nodes Based on Centrality")
plt.show()

# ------------------ Graph 5 ------------------
plt.figure(figsize=(10, 6))
sns.heatmap(centrality_df[["Degree Centrality", "Betweenness Centrality"]].corr(), annot=True, cmap="coolwarm")
plt.title("Centrality Correlation Heatmap")
plt.show()

# ------------------ Optional: Text Column Analysis ------------------
if TEXT_COL in df.columns:
    print(f"\n📝 Text Analysis (Top words in '{TEXT_COL}')")
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))
    words = []

    for text in df[TEXT_COL].dropna():
        tokens = word_tokenize(text.lower())
        words += [word for word in tokens if word.isalpha() and word not in stop_words]

    word_freq = pd.Series(words).value_counts().head(10)
    
    # ------------------ Graph 6 ------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(x=word_freq.values, y=word_freq.index, palette="cubehelix")
    plt.title("Top 10 Keywords from Text Column")
    plt.xlabel("Frequency")
    plt.show()

print("\n✅ Analysis complete. You may change column names at top to fit your dataset.")
