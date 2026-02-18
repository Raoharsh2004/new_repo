import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('malicious_phish.csv')
cv = CountVectorizer(max_features=5000, ngram_range=(1, 1))
features = cv.fit_transform(df['url'])

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(df['type'])

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert to torch tensors
x_train_tensor = torch.tensor(x_train.toarray(), dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple Feedforward Neural Network
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

input_dim = x_train_tensor.shape[1]
num_classes = len(le.classes_)
model = SimpleNN(input_dim, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
batch_size = 256
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(x_train_tensor.size(0))
    for i in range(0, x_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = x_train_tensor[indices].to(device), y_train_tensor[indices].to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(x_test_tensor.to(device))
    _, predicted = torch.max(outputs, 1)
    y_pred = predicted.cpu().numpy()

print("\nCLASSIFICATION REPORT\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("PyTorch NN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
import pickle
with open('pytorch_model.pkl', 'wb') as f:
    pickle.dump({'model_state_dict': model.state_dict(), 'input_dim': input_dim, 'num_classes': num_classes}, f)
print('Model saved to pytorch_model.pkl')
plt.ylabel("Actual")
plt.show()
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

# Save label encoder
with open('labelencoder.pkl', 'wb') as f:
    pickle.dump(le, f)
