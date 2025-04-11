import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def train_xgb(X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
    }

    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=25,
        scoring='accuracy',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    joblib.dump(search, "xgb_model.pkl")

    best_xgb = search.best_estimator_
    y_pred = best_xgb.predict(X_test)

    print(f"XGBoost Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("XGBoost Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (XGBoost)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
