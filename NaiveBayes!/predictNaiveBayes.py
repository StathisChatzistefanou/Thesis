import cv2
import pickle
import pygame
import tkinter as tk
from tkinter import filedialog

# Συνάρτηση Προ-επεξεργασίας Εικόνας
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (30, 30))
    img = img.flatten()
    return img

# Φόρτωση του Εκπαιδευμένου Μοντέλου με χρήση του pickle
model_filename = "trained_naive_bayes_model.pkl"
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)
print("Model loaded successfully!")

# Δημιουργία Παραθύρου Tkinter και επιλογή εικόνας
root = tk.Tk()
root.withdraw()

input_image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.png")])

# 5 Έλεγχος για το εάν έχει επιλεγεί κάποια εικόνα
if input_image_path:
    print("Selected image:", input_image_path)


# Προ-επεξεργασία και Πρόβλεψη της Εικόνας
preprocessed_img = preprocess_image(input_image_path)
predicted_class = loaded_model.predict([preprocessed_img])[0]  # Get the predicted class
print(predicted_class)

# Ορισμός μηνύματος και ηχητικού αποσπάσματος για κάθε περίπτωση κλάσης
class_messages = {
    0: ("Μέγιστη ταχύτητα(20km/h)", "sounds/20kmh.mp3"),
    1: ("Μέγιστη ταχύτητα(30km/h)", "sounds/30kmh.mp3"),
    2: ("Μέγιστη ταχύτητα(50km/h)", "sounds/50kmh.mp3" ),
    3: ("Μέγιστη ταχύτητα(60km/h)", "sounds/60kmh.mp3"),
    4: ("Μέγιστη ταχύτητα(70km/h)", "sounds/70kmh.mp3"),
    5:  ("Μέγιστη ταχύτητα(80km/h)", "sounds/80kmh.mp3"),
    6: ("Τέλος ορίου ταχύτητας που έχει επιβληθεί με απαγορευτική πινακίδα (80km/h)", "sounds/telos80.mp3"),
    7: ("Μέγιστη ταχύτητα(100km/h)", "sounds/100kmh.mp3"),
    8: ("Μέγιστη ταχύτητα(120km/h)", "sounds/120kmh.mp3"),
    9: ("Απαγορεύεται το προσπέρασμα των μηχανοκινήτων οχημάτων, πλήν των διτρόχων μοτοσυκλετών χωρίς κάνιστρο", "sounds/passing.mp3"),
    10: ("Απαγορεύεται στους οδηγούς φορτηγών αυτοκινήτων μεγίστου επιτρεπόμενου βάρους που υπερβαίνει τους 3,5 τόννους να προσπερνούν άλλα οχήματα", "sounds/passing3,5.mp3"),
    11: ("Διασταύρωση με οδό, πάνω στην οποία αυτοί που κινούνται ωφείλουν να παραχωρήσουν προτεραιότητα", "sounds/diastavrosiProteraiotita.mp3"),
    12: ("Οδός προτεραιότητας", "sounds/priorityRoad.mp3"),
    13: ("Υποχρεωτική παραχώρηση προτεραιότητας", "sounds/yield.mp3"),
    14: ("Υποχρεωτική διακοπή πορείας STOP", "sounds/stop.mp3"),
    15: ("Κλειστή οδός για όλα τα οχήματα και πρός τις δύο κατευθύνσεις", "sounds/kleistiOdos.mp3"),
    16: ("Απαγορεύεται η είσοδος στα φορτηγά αυτοκίνητα", "sounds/noEntranceFortiga.mp3"),
    17: ("Απαγορεύεται η είσοδος σε όλα τα οχήματα", "sounds/noEntry.mp3"),
    18: ("Προσοχή άλλοι κίνδυνοι", "sounds/caution.mp3"),
    19: ("Επικίνδυνη αριστερή στροφή", "sounds/dangerLeft.mp3"),
    20: ("Επικίνδυνη δεξιά στροφή", "sounds/dangerRight.mp3"),
    21: ("Επικίνδυνες δύο αντίρροπες ή διαδοχικές στροφές", "sounds/diadoxikesStrofes.mp3"),
    22: ("Επικίνδυνο ανώμαλο οδόστρωμα, σε κακή κατάσταση", "sounds/bumpyRoad.mp3"),
    23: ("Ολισθηρό οδόστρωμα", "sounds/slipperyRoad.mp3"),
    24: ("Επικίνδυνη στένωση οδοστρώματος στην δεξιά πλευρά", "sounds/narrowRight.mp3"),
    25: ("Κίνδυνος λόγω εκτελούμενων εργασιών στην οδό", "sounds/roadWork.mp3"),
    26: ("Προσοχή, κόμβος ή θέση όπου η κυκλοφορία ρυθμίζεται με τρίχρωμη φωτεινή σηματοδότηση", "sounds/trafficSignals.mp3"),
    27: ("Κίνδυνος λόγω διάβασης πεζών", "sounds/pedestrians.mp3"),
    28: ("Κίνδυνος λόγω συχνής κινήσεως παιδιών", "sounds/childCross.mp3"),
    29: ("Κίνδυνος λόγω συχνής εισόδου διαβάσεως ποδηλατιστών", "sounds/bikeCross.mp3"),
    30: ("Κίνδυνος λόγω πάγου", "sounds/Ice.mp3"),
    31: ("Κίνδυνος απο διέλευση άγριων ζώων", "sounds/31.mp3"),
    32: ("Τέλος όλων των τοπικών απαγορεύσεων οι οποίες έχουν επιβληθεί με απαγορευτικές πινακίδες στα κινούμενα οχήματα", "sounds/32.mp3"),
    33: ("Υποχρεωτική κατεύθυνση πορείας με στροφή δεξιά", "sounds/33.mp3"),
    34: ("Υποχρεωτική κατεύθυνση πορείας με στροφή αριστερά", "sounds/34.mp3"),
    35: ("Υποχρεωτική κατεύθυνση πορείας πρός τα εμπρός", "sounds/35.mp3"),
    36: ("Υποχρεωτική κατεύθυνση πορείας εμπρός η δεξιά", "sounds/36.mp3"),
    37: ("Υποχρεωτική κατεύθυνση πορείας εμπρός ή αριστερά", "sounds/37.mp3"),
    38: ("Υποχρεωτική διέλευση μόνο από την δεξιά πλευρά της νησίδας ή του εμποδίου", "sounds/38.mp3"),
    39: ("Υποχρεωτική διέλευση μόνο από την αριστερή πλευρά της νησίδος η του εμποδίου", "sounds/39.mp3"),
    40: ("Κυκλική υποχρεωτική διαδρομή", "sounds/40.mp3"),
    41: ("Τέλος απαγόρευσης προσπεράσματος το οποίο έχει επιβληθεί με απαγορευτική πινακίδα", "sounds/41.mp3"),
    42: ("Τέλος απαγόρευσης προσπεράσματος από φορτηγά αυτοκίνητα που έχει επιβληθεί με απαγορευτική πινακίδα", "sounds/42.mp3")
}


# Εκτύπωση του προβλεπόμενου μηνύματος κλάσης
if predicted_class in class_messages:
    class_message, audio_file = class_messages[predicted_class]
    print("Predicted Class:", class_message)
    # Load and play audio using pygame
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
else:
    print("Unknown class")
