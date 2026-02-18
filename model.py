from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv('malicious_phish.csv')
cv = CountVectorizer()
features = cv.fit_transform(df.text)
features[:5].toarray()
x_train, x_test, y_train, y_test = train_test_split(features, df.type, test_size=0.2)
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
mnb.score(x_test, y_test)
# Classical ML Models

# Metrics
models = {
    "Random Forest": RandomForestClassifier()
}
for name, model in models.items():
    
    print(f"\n🔥 Training {name}...\n")
    
    # Special case for Gaussian NB (needs dense data)
    if name == "Gaussian NB":
        model.fit(x_train.toarray(), y_train)
        train_score = model.score(x_train.toarray(), y_train)
        test_score = model.score(x_test.toarray(), y_test)
        y_pred = model.predict(x_test.toarray())
    else:
        model.fit(x_train, y_train)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
    
    print("Train Accuracy:", train_score)
    print("Test Accuracy:", test_score)
    
    print("\nCLASSIFICATION REPORT\n")
    print(classification_report(
        y_test, y_pred,
        target_names=['df_phish','df_malware','df_deface','df_benign']
    ))
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['df_phish','df_malware','df_deface','df_benign'],
        yticklabels=['df_phish','df_malware','df_deface','df_benign'])
    
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
