import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("tennis_data.csv")

# Preprocess categorical data by converting them to numbers
data['Outlook'] = data['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
data['Temperature'] = data['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
data['Humidity'] = data['Humidity'].map({'High': 0, 'Normal': 1})
data['Windy'] = data['Windy'].map({'False': 0, 'True': 1})
data['PlayTennis'] = data['PlayTennis'].map({'No': 0, 'Yes': 1})

# Handle missing data - columns with no valid values
imputer = SimpleImputer(strategy='most_frequent')

# Apply the imputer on only the columns with missing values
data_imputed = data.copy()

# Apply imputer only to the columns that have NaN values
for col in data_imputed.columns:
    if data_imputed[col].isnull().any():
        data_imputed[col] = imputer.fit_transform(data_imputed[[col]])

# Split data into features and labels
X = data_imputed[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y = data_imputed['PlayTennis']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the Naïve Bayes classifier
model = CategoricalNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Naïve Bayes classifier: {accuracy * 100:.2f}%")

# Print the predictions for the test data
print(f"Predictions: {y_pred}")
