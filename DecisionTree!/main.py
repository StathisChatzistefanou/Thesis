import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Καθορισμός παραμέτρων εικόνας
IMG_HEIGHT = 30
IMG_WIDTH = 30

# Φόρτωση και προ-επεξεργασία δεδομένων
def load_and_preprocess_data(data_dir):
    image_data = []
    image_labels = []

    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            image = cv2.imread(img_path)
            resized_image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
            image_data.append(resized_image)
            image_labels.append(int(category))

    image_data = np.array(image_data)
    image_labels = np.array(image_labels)
    return image_data, image_labels

# Φόρτωση και προ-επεξεργασία δεδομένων
data_dir = r"C:\Users\stathis\Desktop\archive\Train"
X, y = load_and_preprocess_data(data_dir)

# Διαχωρισμός δεδομένων σε σύνολα εκπαίδευσης και επικύρωσης
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Επιπεδοποίηση των εικόνων για το Decision Tree
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Δημιουργία και εκπαίδευση μοντέλου
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_flat, y_train)

# Αξιολόγηση του μοντέλου στο σύνολο επικύρωσης
y_pred = model.predict(X_val_flat)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Αποθήκευση του εκπαιδευμένου μοντέλου
model_filename = "trained_decision_tree_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)
print("Model saved as", model_filename)
