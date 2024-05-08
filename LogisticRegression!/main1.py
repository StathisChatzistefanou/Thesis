import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Ορισμός διαδρομής για το dataset
data_dir = r"C:\Users\stathis\Desktop\archive"

# Ορισμός συνάρτησης φόρτωσης και προ-επεξεργασίας των δεδομένων
def load_data(data_dir):
    images = []
    labels = []

    for label in os.listdir(os.path.join(data_dir, 'Train')):
        path = os.path.join(data_dir, 'Train', label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            images.append(img.flatten())
            labels.append(int(label))

    return np.array(images), np.array(labels)

# Ορισμός διαστάσεων εικόνων
IMG_HEIGHT = 30
IMG_WIDTH = 30

# Φόρτωση και προ-επεξεργασία των δεδομένων
X, y = load_data(data_dir)

# Διαίρεση των δεδομένων σε σύνολα εκπαίδευσης και επικύρωσης
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Αρχικοποίηση και εκπαίδευση του μοντέλου λογιστικής παλινδρόμησης
model = LogisticRegression(max_iter=10)
model.fit(X_train, y_train)

# Αποθήκευση του εκπαιδευμένου μοντέλου με χρήση του pickle
model_filename = "trained_logistic_regression_model1.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)
print("Model saved as", model_filename)








