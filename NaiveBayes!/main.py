import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Ορισμός των παραμέτρων των εικόνων
IMG_HEIGHT = 30
IMG_WIDTH = 30

# Συνάρτηση Φόρτωσης και Προ-επεξεργασίας Δεδομένων
def load_and_preprocess_data(data_dir):
    image_data = []
    labels = []

    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            image = cv2.imread(img_path)
            resized_image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
            image_data.append(resized_image.flatten())
            labels.append(int(category))

    image_data = np.array(image_data)
    labels = np.array(labels)
    return image_data, labels

# Φόρτωση και Προ-επεξεργασία Δεδομένων
data_dir = r"C:\Users\stathis\Desktop\archive\Train"
X, y = load_and_preprocess_data(data_dir)

# Διαχωρισμός δεδομένων σε σύνολα εκπαίδευσης και επικύρωσης
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Δημιουργία και Εκπαίδευση Μοντέλου Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Πρόβλεψη Ετικετών για το Σύνολο Επικύρωσης
y_pred = model.predict(X_val)

# Υπολογισμός Ακρίβειας για την Αξιολόγηση
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Αποθήκευση του Εκπαιδευμένου Μοντέλου με τη χρήση του Pickle
model_filename = "trained_naive_bayes_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)
print("Model saved as", model_filename)
