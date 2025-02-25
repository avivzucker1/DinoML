import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
import joblib
import time
from config import Config

def train_random_forest(X_train, X_test, y_train, y_test):
    # Train and evaluate Random Forest model
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=42
    )
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return rf_classifier

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    # Train and evaluate Gradient Boosting model with optimized parameters
    gb_classifier = GradientBoostingClassifier(
        n_estimators=500,      # More trees for better learning
        learning_rate=0.005,   # Slower learning rate for better generalization
        max_depth=6,          # Control tree depth to prevent overfitting
        min_samples_split=4,  # Minimum samples required to split a node
        subsample=0.9,        # Use 90% of data per tree for robustness
        max_features='sqrt',  # Use sqrt of features for each split
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
        random_state=42
    )
    
    gb_classifier.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = gb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nGradient Boosting Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Print feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': gb_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    
    return gb_classifier

def train_decision_tree(X_train, X_test, y_train, y_test):
    # Train and evaluate Decision Tree model
    dt_classifier = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    dt_classifier.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nDecision Tree Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return dt_classifier

def preprocess_data(df):
    # Clean and preprocess data, handling infinities and extreme values
    
    # Calculate effective speed and derived metrics
    df['effective_speed'] = df[['obstacle_speed', 'obstacle_avg_speed']].max(axis=1)
    df['time_to_impact'] = df['distance_to_obstacle'] / df['effective_speed']
    df['clearance_ratio'] = df['distance_to_obstacle'] / df['obstacle_width']
    
    # Calculate safety margin with speed consideration
    df['safety_margin'] = (df['distance_to_obstacle'] - df['effective_speed'] * 0.7) / df['obstacle_width']
    
    # Calculate time-based features
    df['weighted_time'] = df['time_to_impact'] * (100 / df['effective_speed'])
    
    # Calculate urgency metrics
    df['jump_urgency'] = (df['effective_speed'] ** 2) / (df['distance_to_obstacle'] * 100)
    
    # Handle infinities and extreme values
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        df[column] = df[column].fillna(df[column].median())
        df[column] = np.clip(df[column], 0, 1000)  # Clip extreme values
    
    return df

# Main execution
if __name__ == "__main__":
    # Load and prepare training data
    df = pd.read_csv(os.path.join(Config.SCREENSHOT_FOLDER, Config.TRAINING_RESULTS_FILE))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data
    
    print("Class distribution:")
    print(df['success_or_fail'].value_counts())
    
    # Preprocess features
    df = preprocess_data(df)
    
    # Select features for training
    X = df[Config.TRAINING_HEADERS[:-1]]  # All columns except success_or_fail
    y = df['success_or_fail']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Train all models
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    gb_model = train_gradient_boosting(X_train, X_test, y_train, y_test)
    dt_model = train_decision_tree(X_train, X_test, y_train, y_test)
    
    # Save trained models
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/rf_classifier.joblib')
    joblib.dump(gb_model, 'models/gb_classifier.joblib')
    joblib.dump(dt_model, 'models/dt_classifier.joblib')
    
    # Save model metadata
    model_metadata = {
        'feature_names': X.columns.tolist(),
        'class_names': ['fail', 'success'],
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(model_metadata, 'models/model_metadata.joblib')
    
    print("\nAll models saved to models/ directory")


