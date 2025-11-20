import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import kagglehub
import os

# Re-download dataset to ensure it's available in the current environment
path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")

# Construct the full path to the CSV file
csv_file_path = os.path.join(path, 'Crop_recommendation.csv')

# Load data from the correct path
data = pd.read_csv(csv_file_path)

Y = data['label']
X = data[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]

scaler = MinMaxScaler()
encoder = LabelEncoder()
X_scaled = scaler.fit_transform(X)
Y_scaled = encoder.fit_transform(Y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_scaled,Y_scaled,test_size = 0.2,random_state = 2)

# Instantiate the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(Xtrain, Ytrain)

print("RandomForestClassifier trained successfully.")

from sklearn.metrics import accuracy_score

# Make predictions on the test set
Ypred_rf = rf_model.predict(Xtest)

# Calculate the accuracy score
accuracy_rf = accuracy_score(Ytest, Ypred_rf)

print(f"Random Forest Classifier Accuracy: {accuracy_rf*100:.2f}%")

new_data_sample_rf = Xtest[0:1]
predictions_rf = rf_model.predict(new_data_sample_rf)

# To get the actual crop label, you'll need the inverse transform of the LabelEncoder
predicted_crop_rf = encoder.inverse_transform(predictions_rf)

print(f"Predicted class index by Random Forest: {predictions_rf[0]}")
print(f"Predicted crop by Random Forest: {predicted_crop_rf[0]}")

# Compare with actual label
actual_class_index_rf = Ytest[0]
actual_crop_rf = encoder.inverse_transform([actual_class_index_rf])
print(f"Actual crop: {actual_crop_rf[0]}")

import joblib

# Save the Random Forest model
joblib.dump(rf_model, 'random_forest_crop_recommendation_model.joblib')

print("Random Forest model saved successfully in joblib format.")