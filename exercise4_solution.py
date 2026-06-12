import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# HEART DISEASE
# ==========================================

url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"

heart = pd.read_csv(url)

print(heart.isnull().sum())

heart.fillna(
    heart.mean(numeric_only=True),
    inplace=True
)

corr_matrix = heart.corr()

print(corr_matrix['target'].sort_values(ascending=False))

plt.figure(figsize=(12,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
)

plt.title("Heart Disease Correlation Matrix")
plt.show()

# Boxplots

fig, axes = plt.subplots(1,2, figsize=(12,5))

sns.boxplot(
    x='target',
    y='chol',
    data=heart,
    ax=axes[0]
)

axes[0].set_title("Cholesterol vs Heart Disease")

sns.boxplot(
    x='target',
    y='age',
    data=heart,
    ax=axes[1]
)

axes[1].set_title("Age vs Heart Disease")

plt.show()

# ==========================================
# E-COMMERCE SHIPPING
# ==========================================

shipping = pd.read_csv("Train.csv")

late_delivery = shipping.groupby(
    'Product_importance'
)['Reached.on.Time_Y.N'].mean()

print(late_delivery)

plt.figure(figsize=(10,6))

sns.countplot(
    x='Warehouse_block',
    hue='Product_importance',
    data=shipping
)

plt.title(
    "Delivery Status by Product Importance and Warehouse"
)

plt.show()
