import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text

df = pd.read_csv('datasets/synthetic_dataset.csv')
X = df[['feature_0', 'feature_1', 'feature_2', 'feature_3']]
y = df['class']

# Round features to 1 decimal place to match LLM input format
print("Rounding features to 1 decimal place to match LLM format...")
X = X.round(1)

print("Sample of rounded data:")
print(X.head())
print()

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  

print(f"Train set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

dt = DecisionTreeClassifier(random_state=42, max_depth=5) 
dt.fit(X_train, y_train)

train_pred = dt.predict(X_train)
val_pred = dt.predict(X_val)
test_pred = dt.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
val_accuracy = accuracy_score(y_val, val_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nDecision Tree Rules (trained on 1 decimal precision):")
print("=" * 50)
tree_rules = export_text(dt, feature_names=['feature_0', 'feature_1', 'feature_2', 'feature_3'])
print(tree_rules)

# Verify that the tree thresholds reflect the 1-decimal precision
print("\nVerification: Tree uses thresholds based on 1-decimal data")
print("This ensures fair comparison with LLM that sees 1-decimal inputs")