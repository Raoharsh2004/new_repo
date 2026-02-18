from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import pickle
import re
import os

app = Flask(__name__)
CORS(app)

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Define the Model Architecture
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 3. Initialize and Load Model
# Note: Ensure these files are in the same directory as app.py
try:
    vector = pickle.load(open("vectorizer.pkl", 'rb'))
    le = pickle.load(open("labelencoder.pkl", 'rb'))
    model_data = pickle.load(open("pytorch_model.pkl", "rb"))
    
    input_dim = model_data['input_dim']
    num_classes = model_data['num_classes']
    model = SimpleNN(input_dim, num_classes).to(device)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
except FileNotFoundError as e:
    print(f"Error: Missing model files. {e}")

# API endpoint for frontend JS
@app.route("/api/analyze", methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Preprocessing
    cleaned_url = re.sub(r'^https?://(www\.)?', '', url)
    X = vector.transform([cleaned_url])
    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_label = le.inverse_transform(predicted.cpu().numpy())[0]
    
    # Logic to determine safety based on labels
    # Matching the keys expected by your frontend's 'mlModel' display
    if pred_label.lower() in ['phishing', 'bad', 'malicious']:
        safe = False
    elif pred_label.lower() in ['benign', 'good', 'healthy']:
        safe = True
    else:
        safe = None

    return jsonify({
        "URL": url,
        "mlModel": safe,
        "safe": safe
    })

# Render main page
@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

if __name__ == '__main__':
    # Using port 5000 as defined in your original snippet
    app.run(debug=True, port=5000)