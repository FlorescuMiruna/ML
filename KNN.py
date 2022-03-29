import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from skimage import io
from PIL import Image
from numpy import asarray
from sklearn.naive_bayes import MultinomialNB 


f1 = open("test.txt")
f2 = open("train.txt")
f3 = open("validation.txt")
test_img_paths = f1.read().split()
train_paths = f2.read().split()
train_img_paths = []
train_labels = []
for train_path in train_paths:
    train_img_path, train_label = train_path.split(",")
    train_img_paths.append(train_img_path)
    train_labels.append(int(train_label))


test_images = []
for path in test_img_paths:
    image = (asarray(Image.open('./test/' + path))).flatten().reshape(2500)
    test_images.append(image)

train_images = []
for path in train_img_paths:
    image = (asarray(Image.open('./train/' + path))).flatten().reshape(2500)
    train_images.append(image)






test_images = np.array(test_images)
train_images = np.array(train_images)

print(type(test_images))
print(test_images.shape)
print(test_images)

print(type(train_images))
print(train_images.shape)
print(train_images)

print(train_labels)
train_labels = np.array(train_labels)
print(train_labels)

validation_paths = f3.read().split()
validation_img_paths = []
validation_labels = []

for validation_path in validation_paths:
    validation_img_path, validation_label = validation_path.split(",")
    validation_img_paths.append(validation_img_path)
    validation_labels.append(int(validation_label))

validation_images = []
for path in validation_img_paths:
    image = (asarray(Image.open('./validation/' + path))).flatten().reshape(2500)
    validation_images.append(image)




def acurracy(train_images,train_labels,test_images, test_labels):
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(train_images, train_labels)
    predictions = naive_bayes_model.predict(test_images)
    accuracy = (predictions == test_labels).sum() / len(test_images)
    return accuracy

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(validation_images, validation_labels)
naive_bayes_model.fit(train_images,train_labels)
predictions = naive_bayes_model.predict(test_images)

