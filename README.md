# PREDICTIVE-ANALYTICS-FOR-CLOUD-STORAGE-FAILURE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Step 1: Generate synthetic data and save it to a CSV file
def create_synthetic_data(csv_file_path):
    # Set a random seed for reproducibility
    np.random.seed(42)
    # Generate synthetic data
    num_samples = 1000
    data = {
        'disk_usage': np.random.uniform(0, 100, num_samples),  # Disk usage percentage
        'memory_usage': np.random.uniform(0, 100, num_samples),  # Memory usage percentage
        'cpu_load': np.random.uniform(0, 100, num_samples),  # CPU load percentage
        'network_latency': np.random.uniform(0, 200, num_samples),  # Network latency in ms
        'temperature': np.random.uniform(20, 80, num_samples),  # Temperature in Celsius
        'failure': np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])  # 10% failure rate
    }
    # Create a DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    print(f"Sample CSV file created at: {csv_file_path}")
# Step 2: Load the dataset and handle potential errors
def load_data(csv_file_path):
    try:
        # Attempt to read the CSV file
        data = pd.read_csv(csv_file_path, on_bad_lines='skip', encoding='utf-8')
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
# Step 3: Perform exploratory data analysis (EDA)
def perform_eda(data):
    # Display basic statistics
    print(data.describe())
    # Visualize the distribution of the target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(x='failure', data=data)
    plt.title('Distribution of Target Variable (Failure)')
    plt.xlabel('Failure')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['No Failure', 'Failure'])
    plt.show()
    # Visualize correlations
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
# Step 4: Train a predictive model
def train_model(data):
    # Feature selection
    X = data.drop('failure', axis=1)  # Features
    y = data['failure']  # Target variable
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Model Prediction
    y_pred = model.predict(X_test)
    # Model Evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    # Feature Importance
    feature_importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    # Plotting Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.show()
# Main execution
if __name__ == "__main__":
    # Specify the path for the CSV file
    csv_file_path = 'cloud_storage_data.csv'  # Update this line if needed
    # Create synthetic data and save it to a CSV file
    create_synthetic_data(csv_file_path)
    # Load the dataset
    data = load_data(csv_file_path)
    # Perform exploratory data analysis (EDA)
    if data is not None:
        perform_eda(data)
        # Train a predictive model
        train_model(data)
