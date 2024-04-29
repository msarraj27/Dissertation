import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeRegressor 
import xgboost as xgb

file_input = ""
df = pd.read_csv(file_input)
data = df


#data preprocessing
data.dropna(subset=['Resolution'], inplace=True)
data['Resolution'].fillna('Unknown', inplace=True)

# columns to drop from training set
columns_to_drop = ['Resolution', 'Formatted ID', 'Name', 'Description', 'ServiceNow_Problem']

# Separate features (X) and target variable (y)
X = data.drop(columns_to_drop, axis=1)
y = data['Resolution'] 

# Initialize LabelEncoder for y split
label_encoder = LabelEncoder()

# Encode target variable (y) into numeric labels
y_encoded = label_encoder.fit_transform(y)

# One-hot encode categorical features in X
X_encoded = pd.get_dummies(X)  

$split the data for training
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)


class SimpleClassifier:
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier()

    def fit(self, X, y):
        # Scale input features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the XGBoost classifier
        self.model.fit(X_scaled, y)

    def predict(self, X):
        # Scale input features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions using the XGBoost classifier
        return self.model.predict(X_scaled)

# Example usage:

# Assuming X_train, X_test, y_train, and y_test are defined
# Instantiate the custom XGBoost classifier
model = SimpleClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print(classification_report(y_test, y_pred))


