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

# Merge Transactions, Customers, and Products data
merged_df = transactions_df.merge(customers_df, on="CustomerID").merge(products_df, on="ProductID")

# Create a pivot table for Customer-Product matrix
customer_product_matrix = merged_df.pivot_table(index="CustomerID", columns="ProductID", values="Quantity", aggfunc="sum", fill_value=0)

# Compute cosine similarity
similarity_matrix = pd.DataFrame(cosine_similarity(customer_product_matrix), index=customer_product_matrix.index, columns=customer_product_matrix.index)

# Generate lookalike recommendations
lookalike_results = {}
for customer in similarity_matrix.index[:20]:  # First 20 customers
    similar_customers = similarity_matrix[customer].sort_values(ascending=False)[1:4]  # Top 3 excluding itself
    lookalike_results[customer] = [(idx, round(score, 2)) for idx, score in zip(similar_customers.index, similar_customers.values)]

# Save Lookalike results
lookalike_df = pd.DataFrame({
    "CustomerID": list(lookalike_results.keys()),
    "Lookalikes": [v for v in lookalike_results.values()]
})
lookalike_df.to_csv("surya_k_lookalike.csv", index=False)
print(lookalike_df)

print("Lookalike recommendations saved to 'surya_k_lookalike.csv'")
