import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load Titanic Dataset

titanic = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

# Handle Missing Values

titanic['Age'].fillna(
    titanic['Age'].median(),
    inplace=True
)

titanic['Embarked'].fillna(
    titanic['Embarked'].mode()[0],
    inplace=True
)

# Encode Categorical Variables

titanic['Sex'] = titanic['Sex'].map({
    'male': 0,
    'female': 1
})

titanic = pd.get_dummies(
    titanic,
    columns=['Embarked'],
    drop_first=True
)

# Features and Target

X = titanic[
    [
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked_Q',
        'Embarked_S'
    ]
]

y = titanic['Survived']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("AUC-ROC  :", auc)
