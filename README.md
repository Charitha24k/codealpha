
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('customer_behavior.csv')  # Update filename/path if needed

# Basic information
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# Preview dataset
print("\nFirst 5 Rows:")
print(df.head())

# ---------------------------
# Data Cleaning
# ---------------------------

# Example: Drop rows with missing CustomerID or PurchaseAmount
df = df.dropna(subset=['CustomerID', 'PurchaseAmount'])

# Fill missing Age with median
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)

# Convert dates if available
if 'LastPurchaseDate' in df.columns:
    df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'])

# ---------------------------
# EDA: Univariate Analysis
# ---------------------------

# Plot Age distribution
if 'Age' in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title('Customer Age Distribution')
    plt.show()

# Purchase amount distribution
plt.figure(figsize=(6,4))
sns.histplot(df['PurchaseAmount'], bins=30, kde=True)
plt.title('Purchase Amount Distribution')
plt.show()

# Gender count (if exists)
if 'Gender' in df.columns:
    plt.figure(figsize=(4,4))
    sns.countplot(x='Gender', data=df)
    plt.title('Gender Distribution')
    plt.show()

# ---------------------------
# EDA: Bivariate Analysis
# ---------------------------

# Age vs PurchaseAmount
if 'Age' in df.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='Age', y='PurchaseAmount', data=df)
    plt.title('Age vs Purchase Amount')
    plt.show()

# Gender vs PurchaseAmount
if 'Gender' in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Gender', y='PurchaseAmount', data=df)
    plt.title('Purchase Amount by Gender')
    plt.show()

# ---------------------------
# Time-Based Analysis
# ---------------------------

if 'LastPurchaseDate' in df.columns:
    df['PurchaseMonth'] = df['LastPurchaseDate'].dt.to_period('M')
    monthly_data = df.groupby('PurchaseMonth')['PurchaseAmount'].sum()
    monthly_data.plot(kind='bar', figsize=(10,5), title='Monthly Purchase Trends')
    plt.ylabel('Total Purchase Amount')
    plt.xlabel('Month')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------------------------
# Customer Segmentation
# ---------------------------

# Total spent per customer
customer_spending = df.groupby('CustomerID')['PurchaseAmount'].sum().sort_values(ascending=False)

# Top 10 Customers
top_customers = customer_spending.head(10)
print("Top 10 Customers by Total Spend:")
print(top_customers)

# Plot
top_customers.plot(kind='bar', figsize=(8,4), title='Top 10 Customers by Spend')
plt.ylabel('Total Spend')
plt.show()
