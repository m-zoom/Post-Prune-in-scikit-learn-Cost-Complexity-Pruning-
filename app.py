from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# 1. Realistic sample data
data = {
    "tem": [5, 10, 12, 15, 18, 20, 22, 25, 27, 30, 32, 35, 38],
    "hum": [80, 75, 60, 55, 50, 65, 60, 70, 75, 85, 60, 50, 45],
}
labels = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]  # 0=cold, 1=warm, 2=hot

df = pd.DataFrame(data)
X = df[["tem", "hum"]]
y = labels

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3. Get cost-complexity pruning path
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # exclude the last alpha that prunes all leaves

# 4. Train one model per alpha
clfs = []
for alpha in ccp_alphas:
    model = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    model.fit(X_train, y_train)
    clfs.append(model)

# 5. Evaluate each model
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# 6. Plot accuracy vs alpha
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label="Train Accuracy")
plt.plot(ccp_alphas, test_scores, marker='o', label="Test Accuracy")
plt.xlabel("ccp_alpha (Pruning strength)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Pruning Level")
plt.legend()
plt.grid(True)
plt.show()

# 7. Choose best model (highest test accuracy)
best_index = test_scores.index(max(test_scores))
best_model = clfs[best_index]
print(f"Best alpha: {ccp_alphas[best_index]:.5f}")
print(f"Best test accuracy: {test_scores[best_index]:.4f}")

# 8. Visualize best tree
plt.figure(figsize=(12, 6))
plot_tree(best_model, feature_names=["tem", "hum"], class_names=["Cold", "Warm", "Hot"], filled=True)
plt.title("Best Pruned Decision Tree")
plt.show()
