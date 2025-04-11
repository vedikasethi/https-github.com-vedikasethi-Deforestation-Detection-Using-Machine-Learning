from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def train_svc(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    param_grid = {
        'C': [10, 100, 1000],
        'gamma': [0.01, 0.1, 1]
    }

    svc = SVC(kernel='rbf')
    grid_search = GridSearchCV(svc, param_grid, scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)

    joblib.dump(grid_search, "svc_grid_optimized_model.pkl")
    best_svc = grid_search.best_estimator_
    y_pred = best_svc.predict(X_test_scaled)

    print(f"SVC Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("SVC Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (SVC)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
