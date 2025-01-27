import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# Load datasets
customers_df = pd.read_csv("Customers.csv")
products_df = pd.read_csv("Products.csv")
transactions_df = pd.read_csv("Transactions.csv")

# Merge Transactions with Customers
transactions_with_customers = transactions_df.merge(customers_df, on="CustomerID", how="inner")

# Aggregate customer transaction data
customer_features = transactions_with_customers.groupby("CustomerID").agg({
    "TotalValue": "sum",   # Sum of TotalValue for each customer
    "Quantity": "sum",     # Sum of Quantity for each customer
    "Region": "first"      # Take the first region for each customer (assumes one region per customer)
}).reset_index()

# One-hot encode Region
customer_features = pd.get_dummies(customer_features, columns=["Region"], drop_first=True)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features.iloc[:, 1:])  # Exclude CustomerID for scaling

# Perform clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust the number of clusters as needed
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the DataFrame
customer_features["Cluster"] = clusters

# Evaluate clustering using Davies-Bouldin Index
db_index = davies_bouldin_score(scaled_features, clusters)
print(f"Davies-Bouldin Index: {db_index:.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=clusters, palette="viridis")
plt.title("Customer Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Cluster")
plt.show()

# Save clustering results
customer_features.to_csv("surya_k_clustering.csv", index=False)
print("Clustering results saved to 'surya_k_clustering.csv'")
