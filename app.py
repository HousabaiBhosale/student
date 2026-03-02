from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable Cross-Origin Resource Sharing if UI is run separately

# Load models on startup using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
models = {}

def load_models():
    global models
    try:
        if not os.path.exists(MODELS_DIR):
            return False
            
        new_models = {}
        with open(os.path.join(MODELS_DIR, 'linear_regression.pkl'), 'rb') as f:
            new_models['lr'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'decision_tree.pkl'), 'rb') as f:
            new_models['dt'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'knn.pkl'), 'rb') as f:
            new_models['knn'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'kmeans.pkl'), 'rb') as f:
            new_models['kmeans'] = pickle.load(f)
            
        models = new_models
        print("✅ Models loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        return False

load_models()

@app.route('/')
def index():
    # Serve the base.html file located in the templates/ directory
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Attempt to reload models if memory is empty (happens if trained after server start)
    if not models:
        load_models()
        
    if not models:
        return jsonify({"status": "error", "message": "Models not found on server. Please run train_models.py and restart the server."}), 500

    try:
        data = request.json
        # Expected keys matching JS side:
        # ['att', 'study', 'assign', 'gpa', 'part', 'net', 'sleep', 'fam', 'extra']
        features = [
            float(data.get('att', 0)),
            float(data.get('study', 0)),
            float(data.get('assign', 0)),
            float(data.get('gpa', 0)),
            float(data.get('part', 0)),
            float(data.get('net', 0)),
            float(data.get('sleep', 0)),
            float(data.get('fam', 0)),
            float(data.get('extra', 0))
        ]
        
        # Convert to expected DataFrame format for sklearn
        feature_names = ['Attendance', 'StudyHours', 'AssignScore', 'PrevGPA', 
                         'Participation', 'NetUsage', 'Sleep', 'FamilySupport', 'ExtraCurr']
        df = pd.DataFrame([features], columns=feature_names)
        
        results = {}

        # 1. Linear Regression (Score Prediction)
        if 'lr' in models:
            pred_score = models['lr'].predict(df)[0]
            pred_score = max(0, min(100, round(pred_score))) # Clip 0-100
            results['linear_regression'] = int(pred_score)

        # 2. Decision Tree (Pass/Fail)
        if 'dt' in models:
            dt_model = models['dt']['model']
            dt_enc = models['dt']['encoder']
            pred_pass = dt_model.predict(df)[0]
            results['decision_tree'] = dt_enc.inverse_transform([pred_pass])[0]

        # 3. KNN (Performance Category)
        if 'knn' in models:
            knn_model = models['knn']['model']
            knn_scaler = models['knn']['scaler']
            knn_enc = models['knn']['encoder']
            scaled_features = knn_scaler.transform(df)
            pred_perf = knn_model.predict(scaled_features)[0]
            results['knn'] = knn_enc.inverse_transform([pred_perf])[0]

        # 4. K-Means (Risk Cluster)
        if 'kmeans' in models:
            km_model = models['kmeans']['model']
            km_scaler = models['kmeans']['scaler']
            km_features = models['kmeans']['features']
            
            # K-Means was trained on subset of features (e.g. Attendance, Score, StudyHours)
            # We need the predicted score from LR for this
            if 'linear_regression' in results:
                km_subset = pd.DataFrame([[df['Attendance'][0], results['linear_regression'], df['StudyHours'][0]]], 
                                         columns=km_features)
                scaled_km = km_scaler.transform(km_subset)
                cluster = km_model.predict(scaled_km)[0]
                
                # Map cluster to risk string based on logic (0=High Achiever, 1=Avg, 2=At Risk, 3=High Risk)
                risk_map = {0: 'Low', 1: 'Medium', 2: 'Medium-High', 3: 'High'}
                results['kmeans_cluster'] = int(cluster)
                results['kmeans_risk'] = risk_map.get(cluster, 'Unknown')

        return jsonify({"status": "success", "predictions": results})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/analytics', methods=['GET'])
def get_analytics():
    try:
        csv_path = os.path.join(BASE_DIR, 'data', 'student_data.csv')
        if not os.path.exists(csv_path):
            return jsonify({'status': 'error', 'message': 'Dataset not found'})
            
        df = pd.read_csv(csv_path)
        
        # 1. Score Distribution
        bins = [0, 40, 50, 60, 70, 80, 90, 100]
        labels = ['<40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
        score_dist = pd.cut(df['FinalScore'], bins=bins, labels=labels).value_counts().sort_index().tolist()
        
        # 2. Pass/Fail Ratio
        pass_fail = df['Pass'].value_counts().to_dict()
        
        # 3. Performance Categories
        perf_dist = df['Performance'].value_counts().to_dict()
        
        # 4. Correlation
        numeric_df = df.select_dtypes(include=[np.number])
        corrs = numeric_df.corr()['FinalScore'].drop('FinalScore').abs().sort_values(ascending=False).to_dict()
        
        # 5. Risk Clustering
        risk_dist = df['Risk'].value_counts().to_dict()

        return jsonify({
            'status': 'success',
            'data': {
                'score_distribution': {'labels': labels, 'values': score_dist},
                'pass_fail': pass_fail,
                'performance_categories': perf_dist,
                'correlations': corrs,
                'risk_clusters': risk_dist
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'})
        if file and file.filename.endswith('.csv'):
            save_path = os.path.join(BASE_DIR, 'data', 'student_data.csv')
            file.save(save_path)
            return jsonify({'status': 'success', 'message': 'File uploaded successfully'})
        return jsonify({'status': 'error', 'message': 'Invalid file type'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
