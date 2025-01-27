import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate 

# Load datasets
customers_df = pd.read_csv("Customers.csv")
products_df = pd.read_csv("Products.csv")
transactions_df = pd.read_csv("Transactions.csv")

# Convert TransactionDate to datetime
transactions_df["TransactionDate"] = pd.to_datetime(transactions_df["TransactionDate"])

# Total Revenue
total_revenue = transactions_df["TotalValue"].sum()
print("\n=== Total Revenue ===")
print(f"Total Revenue Generated: ${total_revenue:.2f}")

# Top 5 Customers by Revenue
top_customers = transactions_df.groupby("CustomerID")["TotalValue"].sum().sort_values(ascending=False).head(5)
print("\n=== Top 5 Customers by Revenue ===")
print(tabulate(top_customers.reset_index(), headers=["CustomerID", "Total Revenue"], tablefmt="grid"))

# Most Purchased Products (by Quantity)
top_products = transactions_df.groupby("ProductID")["Quantity"].sum().sort_values(ascending=False).head(5)
print("\n=== Most Purchased Products ===")
print(tabulate(top_products.reset_index(), headers=["ProductID", "Total Quantity"], tablefmt="grid"))

# Monthly Sales Trends
transactions_df["Month"] = transactions_df["TransactionDate"].dt.to_period("M")
monthly_sales = transactions_df.groupby("Month")["TotalValue"].sum()

# Plot Monthly Sales Trends
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind="line", marker="o", title="Monthly Sales Trends")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.grid()
plt.show()

# Revenue by Region
transactions_with_customers = transactions_df.merge(customers_df, on="CustomerID")
revenue_by_region = transactions_with_customers.groupby("Region")["TotalValue"].sum().sort_values(ascending=False)
print("\n=== Revenue by Region ===")
print(tabulate(revenue_by_region.reset_index(), headers=["Region", "Total Revenue"], tablefmt="grid"))
