# zeotap-datasets-analysis
# Data Science Assignment: eCommerce Transactions Dataset

## Overview
This repository contains my solutions to the Data Science assignment involving exploratory data analysis (EDA), predictive modeling, and clustering on an eCommerce Transactions dataset. The goal is to derive actionable insights, build a lookalike recommendation model, and perform customer segmentation.

## Dataset
The dataset includes three files:
1. **Customers.csv**: Contains customer details like ID, name, region, and signup date.
2. **Products.csv**: Includes product information such as ID, name, category, and price.
3. **Transactions.csv**: Details transactions with attributes like transaction ID, customer ID, product ID, quantity, price, and total value.

## Tasks and Deliverables

### 1. Exploratory Data Analysis (EDA) and Business Insights
- **Objective**: Analyze the data and extract at least 5 business insights.
- **Deliverables**:
  - `surya_k_EDA.py`: Python file with EDA code.
  - `surya_k_EDA.pdf`: Report containing business insights.

### 2. Lookalike Model
- **Objective**: Build a recommendation model to identify 3 similar customers based on profile and transaction history.
- **Deliverables**:
  - `surya_k_lookalike.py`: Python file with the lookalike model development.
  - `surya_k_lookalike.csv`: File mapping customer IDs to their top 3 lookalikes and similarity scores.

### 3. Customer Segmentation / Clustering
- **Objective**: Perform customer clustering using profile and transaction data.
- **Deliverables**:
  - `surya_k_clustering.py`: Python file with clustering code.
  - `surya_k_clustering.pdf`: Report on clustering results, including:
    - Number of clusters.
    - DB Index value.
    - Visualization of clusters.
  - `surya_k_clustering.csv`: File with clustering results.


## Instructions to Run Code
1. Clone the repository:
   ```bash
   git clone https://github.com/SuryaCodez/zeotap-datasets-analysis
   cd zeotap-datasets-analysis
