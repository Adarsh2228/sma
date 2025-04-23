import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# === CONFIGURATION: Change these column names as per your dataset ===
date_col = "date"  # e.g., 'year', 'month', 'created_at'
value_cols = ["likes", "shares", "comments"]  # any columns to track trends

# === Load Dataset ===
df = pd.read_csv("social_media_time_data.csv")  # replace with your CSV
df = df[[date_col] + value_cols].dropna()

# === Convert Date Column ===
# If it's year or month in string form, normalize it
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])  # Drop rows with invalid dates
df = df.sort_values(by=date_col)

# Extract year/month for trend aggregation
df['Year'] = df[date_col].dt.year
df['Month'] = df[date_col].dt.to_period('M').astype(str)

# === EDA ===
print("Basic Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# === 1. Time vs Count Plot for Each Column ===
plt.figure(figsize=(12, 6))
for col in value_cols:
    plt.plot(df[date_col], df[col], label=col)
plt.title("Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 2. Monthly Aggregated Trend ===
monthly_df = df.groupby('Month')[value_cols].sum().reset_index()
plt.figure(figsize=(12, 6))
for col in value_cols:
    plt.plot(monthly_df['Month'], monthly_df[col], marker='o', label=col)
plt.title("Monthly Aggregated Trend")
plt.xlabel("Month")
plt.ylabel("Total Value")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# === 3. Year-wise Distribution Boxplot ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Year', y=value_cols[0])
plt.title(f"Distribution of {value_cols[0]} by Year")
plt.show()

# === 4. Correlation Heatmap ===
plt.figure(figsize=(8, 5))
sns.heatmap(df[value_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Metrics")
plt.show()

# === 5. Interactive Trend Chart (Plotly) ===
fig = px.line(df, x=date_col, y=value_cols, title="Interactive Time Series Trends")
fig.show()

# === Optional: Yearly Mean Table for Reporting ===
yearly_mean = df.groupby("Year")[value_cols].mean()
print("\nYearly Averages:")
print(yearly_mean)
