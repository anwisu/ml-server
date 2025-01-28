import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
# Expanded Sample data with multiple students
data = {
    'Previous_1': [0, 1, 2, 2],  # MA=0, MO=1, MI=2, I=3
    'Previous_2': [0, 0, 2, 2],  # Input: Q1 and Q2 levels
    'Target': [1, 1, 2, 2]  # Target: Q3 prediction
}
df = pd.DataFrame(data)
# Split data (training on multiple students)
X = df[['Previous_1', 'Previous_2']]  # Features (previous evaluation levels)
y = df['Target']  # Labels (target level for the next evaluation)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


joblib.dump(model, "domain4.pkl");