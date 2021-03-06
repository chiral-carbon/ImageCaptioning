# Using the VGG16 network to extract features from each image. 
# The resulting feature file will have features of each image as a 1D, 4096 element array

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from pickle import dump
from os import listdir
from tqdm import tqdm
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

# Defining a function to extract features from images stored in a given directory
def extract(dirname):

	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	print(model.summary())

	features = dict()

	for name in tqdm(listdir(dirname)):
		filename = dirname + '/' + name
		image = load_img(filename, target_size=(224, 224))
		image = img_to_array(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)
		feature = model.predict(image, verbose=0)
		image_id = name.split('.')[0]
		features[image_id] = feature

	return features

directory_name = filedialog.askdirectory() # Open the Flickr8k_Dataset directory containing all the images
features = extract(directory_name)

print("Extracted Features: ", len(features))

# dumping features into a pickled file
dump(features, open('features.pkl', 'wb'))

