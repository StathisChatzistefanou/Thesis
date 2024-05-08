import tensorflow as tf
print(tf.__version__)
import numpy as np
import os
import cv2
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


np.random.seed(42)

from matplotlib import style

style.use('fivethirtyeight')

## ορισμός διαδρομής των δεδομένων
data_dir = r"C:\Users\stathis\Desktop\archive"
train_path = r"C:\Users\stathis\Desktop\archive\Train"
test_path = r"C:\Users\stathis\Desktop\archive\Test"

## Αλλαγή μεγέθους εικόνων σε 30x30x3
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3


## Ανίχνευση κάθε κλάσης
NUM_CATEGORIES = len(os.listdir(train_path))
NUM_CATEGORIES

# Δημιουργία ετικετών
classes = {0: 'Μέγιστη ταχύτητα(20km/h)',
           1: 'Μέγιστη ταχύτητα(30km/h)',
           2: 'Μέγιστη ταχύτητα(50km/h)',
           3: 'Μέγιστη ταχύτητα(60km/h)',
           4: 'Μέγιστη ταχύτητα(70km/h)',
           5: 'Μέγιστη ταχύτητα(80km/h)',
           6: 'Τέλος ορίου ταχύτητας που έχει επιβληθεί με απαγορευτική πινακίδα (80km/h)',
           7: 'Μέγιστη ταχύτητα(100km/h)',
           8: 'Μέγιστη ταχύτητα(120km/h)',
           9: 'Απαγορεύεται το προσπέρασμα των μηχανοκινήτων οχημάτων, πλήν των διτρόχων μοτοσυκλετών χωρίς κάνιστρο',
           10: 'Απαγορεύεται στους οδηγούς φορτηγών αυτοκινήτων μεγίστου επιτρεπόμενου βάρους που υπερβαίνει τους 3,5 τόννους να προσπερνούν άλλα οχήματα',
           11: 'Διασταύρωση με οδό, πάνω στην οποία αυτοί που κινούνται ωφείλουν να παραχωρήσουν προτεραιότητα',
           12: 'Οδός προτεραιότητας',
           13: 'Υποχρεωτική παραχώρηση προτεραιότητας',
           14: 'Υποχρεωτική διακοπή πορείας STOP',
           15: 'Κλειστή οδός για όλα τα οχήματα και πρός τις δύο κατευθύνσεις',
           16: 'Απαγορεύεται η είσοδος στα φορτηγά αυτοκίνητα',
           17: 'Απαγορεύεται η είσοδος σε όλα τα οχήματα',
           18: 'Προσοχή άλλοι κίνδυνοι',
           19: 'Επικίνδυνη αριστερή στροφή',
           20: 'Επικίνδυνη δεξιά στροφή',
           21: 'Επικίνδυνες δύο αντίρροπες ή διαδοχικές στροφές',
           22: 'Επικίνδυνο ανώμαλο οδόστρωμα, σε κακή κατάσταση',
           23: 'Ολισθηρό οδόστρωμα',
           24: 'Επικίνδυνη στένωση οδοστρώματος στην δεξιά πλευρά',
           25: 'Κίνδυνος λόγω εκτελούμενων εργασιών στην οδό',
           26: 'Προσοχή, κόμβος ή θέση όπου η κυκλοφορία ρυθμίζεται με τρίχρωμη φωτεινή σηματοδότηση',
           27: 'Κίνδυνος λόγω διάβασης πεζών',
           28: 'Κίνδυνος λόγω συχνής κινήσεως παιδιών',
           29: 'Κίνδυνος λόγω συχνής εισόδου διαβάσεως ποδηλατιστών',
           30: 'Κίνδυνος λόγω πάγου',
           31: 'Κίνδυνος απο διέλευση άγριων ζώων',
           32: 'Τέλος όλων των τοπικών απαγορεύσεων οι οποίες έχουν επιβληθεί με απαγορευτικές πινακίδες στα κινούμενα οχήματα',
           33: 'Υποχρεωτική κατεύθυνση πορείας με στροφή δεξιά',
           34: 'Υποχρεωτική κατεύθυνση πορείας με στροφή αριστερά',
           35: 'Υποχρεωτική κατεύθυνση πορείας πρός τα εμπρός',
           36: 'Υποχρεωτική κατεύθυνση πορείας εμπρός η δεξιά',
           37: 'Υποχρεωτική κατεύθυνση πορείας εμπρός ή αριστερά',
           38: 'Υποχρεωτική διέλευση μόνο από την δεξιά πλευρά της νησίδας ή του εμποδίου',
           39: 'Υποχρεωτική διέλευση μόνο από την αριστερή πλευρά της νησίδος η του εμποδίου',
           40: 'Κυκλική υποχρεωτική διαδρομή',
           41: 'Τέλος απαγόρευσης προσπεράσματος το οποίο έχει επιβληθεί με απαγορευτική πινακίδα',
           42: 'Τέλος απαγόρευσης προσπεράσματος από φορτηγά αυτοκίνητα που έχει επιβληθεί με απαγορευτική πινακίδα'}


## Συλλογή των δεδομένων εκπαίδευσης
image_data = []
image_labels = []

for i in range(NUM_CATEGORIES):
    path = data_dir + '/Train/' + str(i)
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + img)

## Αλλαγή της λίστας σε πίνακα numpy
image_data = np.array(image_data)
image_labels = np.array(image_labels)

print(image_data.shape, image_labels.shape)

## Ανακάτεμα των δεδομένων εκπαίδευσης
shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]

## Διαχωρισμός των δεδομένων σε σύνολο εκπαίδευσης και επικύρωσης
X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42,
                                                  shuffle=True)

X_train = X_train / 255
X_val = X_val / 255

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_val.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_val.shape)

## Kωδικοποίηση των ετικετών
y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)

print(y_train.shape)
print(y_val.shape)


## Κατασκευή του μοντέλου
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, channels)),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),

    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.5),

    keras.layers.Dense(43, activation='softmax')
])

from tensorflow.keras.optimizers import legacy as legacy_optimizers

lr = 0.001
epochs = 1
decay = lr / (epochs * 0.5)

opt = legacy_optimizers.Adam(lr=lr, decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

## Συμπλήρωση των δεδομένων και εκπαίδευση
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

history = model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_val, y_val))
model.save("testing1.h5")



