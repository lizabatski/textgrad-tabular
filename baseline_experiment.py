import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Classic models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load and shuffle
df = pd.read_csv("datasets/Iris.csv")
random.seed(42)


X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"].str.split("-").str[-1].str.lower()


le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split 60:20:20
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# define models to test
models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel='rbf', probability=True)
}

# evaluate each model
print("=== Baseline Model Results ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    
    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"{name}")
    print(f"  Validation Accuracy: {val_acc:.2%}")
    print(f"  Test Accuracy:       {test_acc:.2%}")
    print("-" * 40)
