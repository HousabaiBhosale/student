import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score, silhouette_score

def train_and_save_models():
    print("Loading dataset...")
    data_path = 'data/student_data.csv'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Run generate_data.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Input features
    features = ['Attendance', 'StudyHours', 'AssignScore', 'PrevGPA', 
                'Participation', 'NetUsage', 'Sleep', 'FamilySupport', 'ExtraCurr']
    X = df[features]
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("--- 1. Linear Regression (Final Score) ---")
    y_score = df['FinalScore']
    lr = LinearRegression()
    lr.fit(X, y_score)
    preds = lr.predict(X)
    print(f"R2 Score (Train): {r2_score(y_score, preds):.4f}")
    with open('models/linear_regression.pkl', 'wb') as f:
        pickle.dump(lr, f)

    print("\n--- 2. Decision Tree (Pass/Fail) ---")
    le_pass = LabelEncoder()
    y_pass = le_pass.fit_transform(df['Pass']) # 0 or 1
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X, y_pass)
    preds = dt.predict(X)
    print(f"Accuracy (Train): {accuracy_score(y_pass, preds):.4f}")
    with open('models/decision_tree.pkl', 'wb') as f:
        pickle.dump({'model': dt, 'encoder': le_pass}, f)

    print("\n--- 3. K-Nearest Neighbors (Performance Category) ---")
    le_perf = LabelEncoder()
    y_perf = le_perf.fit_transform(df['Performance'])
    # KNN benefits from scaling
    scaler_knn = StandardScaler()
    X_scaled = scaler_knn.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y_perf)
    preds = knn.predict(X_scaled)
    print(f"Accuracy (Train): {accuracy_score(y_perf, preds):.4f}")
    with open('models/knn.pkl', 'wb') as f:
        pickle.dump({'model': knn, 'scaler': scaler_knn, 'encoder': le_perf}, f)

    print("\n--- 4. K-Means (Academic Risk Clusters) ---")
    # Using specific features for clustering risk
    cluster_features = ['Attendance', 'FinalScore', 'StudyHours']
    X_cluster = df[cluster_features]
    scaler_km = StandardScaler()
    X_c_scaled = scaler_km.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_c_scaled)
    sil = silhouette_score(X_c_scaled, clusters)
    print(f"Silhouette Score: {sil:.4f}")
    
    with open('models/kmeans.pkl', 'wb') as f:
        pickle.dump({'model': kmeans, 'scaler': scaler_km, 'features': cluster_features}, f)
        
    print("\nAll models trained and saved to /models directory")

if __name__ == '__main__':
    train_and_save_models()
