from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_loader import load_tiff_files
from svc_model import train_svc
from xgb_model import train_xgb

# Load data
X, y = load_tiff_files("Dataset", target_shape=(64, 64))
print(f"Loaded dataset: {X.shape}, Labels: {y.shape}")

# Shuffle & split
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
train_svc(X_train, y_train, X_test, y_test)
train_xgb(X_train, y_train, X_test, y_test)
