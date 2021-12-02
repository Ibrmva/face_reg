from posix import listdir
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seabron as sns
import cv2
import os
import warnings 
warnings.filterwarnings("ignore")
%matplotlib inline
sns.set(color_codes = True)
train_directory = "train/"
train_dir = os.listdir(train_directory)
classes = []

for folder in train_dir:
    classes.append(folder)
    print(folder)

train_counts = []

for folder in train_dir:
    class_path = train_directory + folder + "/"
    list_train = []
    count = 0 
    for file in os.listdir(class_path):
        count += 1

    train_counts.append(count)

train_counts

#big graph :)
plt.bar(classes, train_counts, width = 0.5)
plt.title("Bar Graph of Train Data")
plt.xlabel("Classes")
plt.ylabel("Counts")

#Scatter plot :)
plt.scatter(classes, train_counts)
plt.plot(train_classes, '-o')
plt.show()

#displot
x = np.array(train_counts)
sns.displot(x, kde = True, bins = 5 )
sns.displot(x)

#reading analyzing the images
for folder in train_dir:
    class_path = train_directory + folder + "/"
    for file in os.listdir(class_path):
        if file.endswith(".jpg"):
            final_path = class_path + file
            print(final_path)
            img = cv2.imread(final_path, cv2.IMREAD_UNCHANGED)
            break
        break
plt.imshow(img)

heigth, width = img.shape

print("""
The Height of the image = {0}
The Width of the image = {1}
The Number of channels = {2}""".format(heigth, width, 1))

image_path = []

for folder in train_dir:
    class_path = train_directory + folder + "/"
    for file in os.listdir(class_path):
        if file.endwith(".jpg"):
            final_path = class_path + file
            image_path.append(final_path)
            break
image_paths 

import matplotlib.image as mpimg

for file in image_paths:
    image = mpimg.imread(file)
    plt.figure()
    pos1 = file.find('/')
    pos2 = file.find('/0')
    title = file[pos1 + 1:pos2]
    plt.title(title, fontsize = 20)
    plt.imshow(image)

#analysis of Validation data:

val_directory = "validation/"
val_dir = os.listdir(val_directory)
classes = []

for folder in train_dir:
    classes.append(folder)
    print(folder)

val_counts = []

for folder in val_dir:
    class_path = val_directory + folder + "/"
    list_train = []
    count = 0 
    for file in oc.listdir(class_path):
        count += 1
    val_counts.append(count)

val_counts

#Bar graph

plt.bar(classes, var_counts, width = 0.5)
plt.title("Big Graph of Validation Data ")
plt.xlabel("Classes")
plt.ylabel("Counts")

#Scatter Plot
plt.scatter(classes, val_counts)
plt.plot(classes, val_counts, '-o')
plt.show()

#Reading and Analyzing the validation images
for folder in val_dir:
    class_path = val_directory + folder + "/"
    for file in os.listdir(class_path):
        if file.endswith(".jpg"):
            final_path = class_path + file
            print("final_path, cv2.IMREAD_UNCHANGED")
            break
        break
plt.imshow(img)

heigt, width = img.shape
print("""
The Height of the image = {0}
The Width of the image = {1}
The Number of channels = {2}""".format(heigth, width, 1))

image_path = []

for folder in val_dir:
    class_path = val_directory + folder + "/"
    for file in os.listdir(class_path):
        if file.endswith(".jpg"):
            final_path = class_path + file
            image_path.append(final_path)
        break
image_path

import matplotlib.image as mping 

for file in image_path:
    image = mping.imread(file)
    plt.figure()
    pos1 = file.find('/')
    pos2 = file.find('/0')
    title = file[pos1 + 1:pos2]
    plt.title(title, fontsize = 20 )
    plt.show(image)

#Gesture Dataset Analysis
train_directory = "train/"
train_dir = os.listdir(train_directory)
classes = []

for folder in train_dir:
    classes.append(folder)
    print(folder)

train_counts = []

for folder in train_dir:
    class_path = train_directory + folder + "/"
    list_train = []
    count = 0 
    for file in os.listdir(class_path):
        count += 1
    train_counts.append(count)

train_counts

#Bar graph
plt.bar(classes, train_counts, width = -0.5)
plt.title("Big graph of Train Data")
plt.xlabel("Classes")
plt.ylabel("Counts")

#Scatter PLot
plt.scatter(classes, train_counts, "-o")
plt.plot(classes, train_counts, '-o')
plt.show()

#Reading and Analyzing the images
for folder in train_dir:
    class_path = train_directory + folder + "/"
    for file in os.listdir(class_path):
        if file.endswith(".jpg"):
            final_path = class_path + file 
            print(final_path)
            img = cv2.imread(final_path, cv2.IMREAD_UNCHANGED)
            break
        break
train1/loser/L1.jpg
plt.imshow(img)

heigth, width, channels = img.shape

print("""
The Heigth of the image = {0}
The Width of the image = {1}
The Number of channels = {2}""".format(heigth, width, channels))

image_path = []:

for folder in train_dir:
    class_path = train_directory + folder +"/"
    for file in os.listdir(class_path):
        if file.endswith(".jpg"):
            final_path = class_path + file
            image_path.append(final_path)
            break
image_path

import matplotlib.image as mpimg

for file in image_paths:
    image = mping.imread(file)
    plt.figure()
    pos1 = file.find('/')
    pos2 = file.find('/0')
    title = file[pos1 + 1 : pos2]
    plt.title(title, fontsize = 20)
    plt.imshow(image)

#Analysis of Validation data
val_directory = "validation1/"
val_dir = os.listdir(val_directory)
classes = []

for folder in train_dir:
    classes.append(folder)
    print(folder)

val_counts = []

for folder in val_dir:
    class_path = val_directory + folder + "/"
    list_train = []
    count = 0
    for file in os.listdir(class_path):
        count += 1

    val_counts.append(count) 

val_counts
