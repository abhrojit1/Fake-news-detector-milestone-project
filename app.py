"""
Flask Backend for Fake News Detection
Trains model once, saves it, and loads on subsequent runs
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os
import numpy as np
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'fake_news_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
STATS_PATH = 'model_stats.pkl'

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 2), 
            min_df=2, 
            max_df=0.8
        )
        self.model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        self.is_trained = False
        self.stats = {}
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\@\w+|\#', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text
    
    def train(self, fake_path, real_path):
        """Train the model on dataset"""
        print("Loading datasets...")
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        
        # Clean data
        fake_df = fake_df.dropna(subset=['title', 'text'])
        real_df = real_df.dropna(subset=['title', 'text'])
        
        # Combine title and text
        fake_df['combined_text'] = fake_df['title'].astype(str) + ' ' + fake_df['text'].astype(str)
        real_df['combined_text'] = real_df['title'].astype(str) + ' ' + real_df['text'].astype(str)
        
        # Add labels
        fake_df['label'] = 1
        real_df['label'] = 0
        
        # Combine and shuffle
        combined_df = pd.concat([fake_df, real_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Total articles: {len(combined_df)}")
        print(f"Fake: {len(fake_df)}, Real: {len(real_df)}")
        
        # Clean text
        print("Cleaning text...")
        texts = combined_df['combined_text'].tolist()
        labels = combined_df['label'].tolist()
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize
        print("Vectorizing...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True
        
        # Evaluate
        print("Evaluating...")
        predictions = self.model.predict(X_test_vec)
        
        self.stats = {
            'accuracy': float(accuracy_score(y_test, predictions) * 100),
            'precision': float(precision_score(y_test, predictions) * 100),
            'recall': float(recall_score(y_test, predictions) * 100),
            'f1_score': float(f1_score(y_test, predictions) * 100),
            'total_articles': len(combined_df),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {self.stats['accuracy']:.2f}%")
        print(f"Precision: {self.stats['precision']:.2f}%")
        print(f"Recall: {self.stats['recall']:.2f}%")
        print(f"F1-Score: {self.stats['f1_score']:.2f}%")
        
        return self.stats
    
    def predict(self, text):
        """Predict if article is fake or real"""
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        
        cleaned = self.clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]
        
        result = {
            'prediction': 'FAKE NEWS' if prediction == 1 else 'REAL NEWS',
            'confidence': float(max(probabilities)) * 100,
            'real_probability': float(probabilities[0]) * 100,
            'fake_probability': float(probabilities[1]) * 100
        }
        
        return result
    
    def save(self):
        """Save model, vectorizer, and stats"""
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(STATS_PATH, 'wb') as f:
            pickle.dump(self.stats, f)
        print("Model saved successfully!")
    
    def load(self):
        """Load model, vectorizer, and stats"""
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(STATS_PATH, 'rb') as f:
                self.stats = pickle.load(f)
            self.is_trained = True
            print("Model loaded successfully!")
            return True
        return False


# Initialize detector
detector = FakeNewsDetector()

# Try to load existing model, or train new one
if not detector.load():
    print("No saved model found. Training new model...")
    print("Please provide paths to Fake.csv and True.csv")
    
    # UPDATE THESE PATHS TO YOUR CSV FILES
    FAKE_PATH = "C:\\Users\\abhrojit bhattachary\\Documents\\Fake.csv" # Change this
    REAL_PATH = "C:\\Users\\abhrojit bhattachary\\Documents\\True.csv"  # Change this
    
    if os.path.exists(FAKE_PATH) and os.path.exists(REAL_PATH):
        detector.train(FAKE_PATH, REAL_PATH)
        detector.save()
    else:
        print("ERROR: CSV files not found! Please update FAKE_PATH and REAL_PATH in the code.")
        print("The API will run but predictions won't work until model is trained.")


@app.route('/')
def index():
    return """
    <h1>Fake News Detection API</h1>
    <p>API is running!</p>
    <ul>
        <li>POST /api/predict - Analyze article</li>
        <li>GET /api/stats - Get model statistics</li>
    </ul>
    """

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    if detector.is_trained:
        return jsonify(detector.stats)
    else:
        return jsonify({'error': 'Model not trained yet'}), 503

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if article is fake or real"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not detector.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 503
        
        result = detector.predict(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Retrain the model (admin only)"""
    try:
        data = request.get_json()
        fake_path = data.get('fake_path', 'Fake.csv')
        real_path = data.get('real_path', 'True.csv')
        
        stats = detector.train(fake_path, real_path)
        detector.save()
        
        return jsonify({
            'message': 'Model retrained successfully',
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FAKE NEWS DETECTION API SERVER")
    print("="*80)
    print(f"Model trained: {detector.is_trained}")
    if detector.is_trained:
        print(f"Accuracy: {detector.stats.get('accuracy', 0):.2f}%")
    print("\nStarting Flask server on http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)