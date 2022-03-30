import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
import os
from PIL import Image
from skimage import io
from PIL import Image
from numpy import asarray


f1 = open("test.txt")
f2 = open("train.txt")
f3 = open("validation.txt")

# TEST
test_img_paths = f1.read().split()  # salvam denumirile pozelor pentru a le putea incarca apoi din folder
test_images = []
for path in test_img_paths:
    image = (asarray(Image.open('./test/' + path))).flatten().reshape(2500) # Luam fiecare poza in parte din folder, folosindu-ne de numele acestora salvate in test_img_paths si ii dam un reshape la 2500 deoarece aceats avea initial 50 * 50
    test_images.append(image)


test_images = np.array(test_images) # Transformam lista de poze intr-un np-array


# print(test_images.shape)
# print(test_images)

# TRAIN

train_paths = f2.read().split() # Citim informatiile din fisier
train_img_paths = []
train_labels = []
for train_path in train_paths:
    train_img_path, train_label = train_path.split(",") # In train_img_path va ramane salvata denumirea pozei, iar in train_label va fi chiar label-ul pozei
    train_img_paths.append(train_img_path) # Adaugam denumirea imaginii in lista de denumiri
    train_labels.append(int(train_label)) # Adaugam label-ul in lista de labeluri pentru train

#print(train_labels)

train_images = []
for path in train_img_paths:
    image = (asarray(Image.open('./train/' + path))).flatten().reshape(2500) # Luam fiecare imagini din folder si o redimensionam la 2500, aceasta fiind initial de 50*50
    train_images.append(image)



# VALIDATION

# In acelasi mod in care am incarcat imaginile si label-urile de antrenare, le incarcam si pe cele de validare
validation_paths = f3.read().split()
validation_img_paths = []
validation_labels = []

for validation_path in validation_paths:
    validation_img_path, validation_label = validation_path.split(",")
    validation_img_paths.append(validation_img_path)
    validation_labels.append(int(validation_label))
    train_labels.append(int(validation_label))

validation_images = []
for path in validation_img_paths:
    image = (asarray(Image.open('./validation/' + path))).flatten().reshape(2500)
    validation_images.append(image)
    train_images.append(image)


train_images = np.array(train_images)
train_labels = np.array(train_labels)


print(type(train_images))
print(train_images.shape)
print(train_images)
print(train_labels)


class KnnClassifier:
    """ Construiesc un model de tip Knn. Acesta se bazeaza pe faptul ca daca un majoritatea
    vecinilor  unui element sunt de un anumit tip, atunci si acesta este de acelasi tip"""
    def __init__(self, train_images, train_labels): # Constructori pentru initializarea imaginilor de train si a label-urilor
        self.train_images = train_images
        self.train_labels = train_labels
    def classify(self, image, neighbours_number):
        """Primesc ca argument numarul de vecini pe care vreau sa il folosesc in modelul meu.
        Calculez distantele pana la vecini, folosindu-ma de imaginile de training si de imaginea
        pe care vreau sa o clasific.Ca formula pentru distanta am folosit radical din suma patratelor
        diferentelor"""
        distances = np.sqrt(np.sum((self.train_images - image)**2, axis=1))
        args = np.argsort(distances)[:neighbours_number] # Sortez distantele si iau argumentele celor mai apropiati vecini
        neighbours_labels = self.train_labels[args]  # Salvez label-urile primilor vecini
        label = np.bincount(neighbours_labels).argmax()  # Selectez vecinul care a aparut de cele mai multe ori
        return label

    def acurracy(self, test_images, test_labels, neighbours_number):
        """ Pentru a calcula acuratetea clasific fiecare imagine folosindu-ma de modelul meu, iar daca
        predictia este aceeasi cu labelul real al imaginii, cresc numarul de predictii corecte
        Acuratetea reprezinta numarul de predictii corecte supra numarul total de imagini testate"""

        predictions = []
        correct_predictions = 0
        for i in range(0, len(test_images)):
            print(neighbours_number, i)
            pred = self.classify(test_images[i], neighbours_number)
            if pred == test_labels[i]:
                correct_predictions += 1

        accuracy = correct_predictions / len(test_images)
        return accuracy

    def display_confusion_matrix(self, test_images, test_labels, neighbours_number):
        predictions = []
        for i in range(0, len(test_images)):
            #print(neighbours_number, i)
            pred = self.classify(test_images[i], neighbours_number)
            predictions.append(pred)
        predictions = np.array(predictions)

        my_confusion_matrix = confusion_matrix(test_labels, predictions, labels=[0,1,2])
        print("~~~~~~~~~ Confusion Matrix ~~~~~~~~~")
        print(my_confusion_matrix)

predictions = []

KNN = KnnClassifier(train_images, train_labels) # Initializez modelul cu datele de training

for i in range(len(test_images)):
    print(i)
    predictions.append(KNN.classify(test_images[i], 3)) # clasific fiecare imagine folosindu-ma de model
    # Pentru fiecare imagine am calculat pentru 3 vecini
print(predictions)




g = open("submission.txt", "w")
sir_afisare = ""

sir_afisare += "id,label\n"
for i in range(len(predictions)):
    sir_afisare += test_img_paths[i] + "," + str(predictions[i]) + "\n"

g.write(sir_afisare)


acuratete = KNN.acurracy( validation_images, validation_labels, 3)
print("Acuratete: ", acuratete)

KNN.display_confusion_matrix( validation_images, validation_labels, 3)
# [[1496 4 0]
#  [ 21 1432 47]
#  [ 23 41 1436]]



# vecini = [1,3,5,7]
# x = []
# for i in vecini:
#     acc = KNN.acurracy(validation_images,validation_labels,i)
#     x.append(acc)
#
# plt.plot(x)
# plt.show()