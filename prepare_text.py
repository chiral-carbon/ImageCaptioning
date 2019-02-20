# Preparing the text dataset for storing the captions with indices as clean descriptions
# and also
# Grabbing the vocabulary from the captions and storing all the words in the vocabulary

import string
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

def load_descriptions(filename):
	file = open(filename, 'r')
	doc = file.read()
	file.close()

	mapping = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		if len(line) < 2:
			continue

		image_id, image_desc = tokens[0], tokens[1:]
		image_id = image_id.split('.')[0]
		image_desc = ' '.join(image_desc)

		if image_id not in mapping:
			mapping[image_id] = list()

		mapping[image_id].append(image_desc)

	return mapping

def clean_descriptions(descriptions):
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			desc = desc.split()
			desc = [word.lower() for word in desc]
			desc = [w.translate(table) for w in desc]
			desc = [word for word in desc if len(word)>1]
			desc = [word for word in desc if word.isalpha()]
			desc_list[i] = ' '.join(desc)

def to_vocabulary(descriptions):
	vocab = set()
	for key in descriptions.keys():
		[vocab.update(d.split()) for d in descriptions[key]]
	return vocab

def save_vocabulary(vocab, filename):
	file = open(filename, 'w')
	for word in vocab:
		file.write(word+"\n")
	file.close()

def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)

	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

filename = filedialog.askopenfilename() # Open the Flickr8k.token.txt file in the Flickr8k_text directory 
# filename = "C:/Users/Abhipsha Das/Desktop/Flickr8k_text/Flickr8k.token.txt"
descriptions = load_descriptions(filename)
print("Loaded: ", len(descriptions))

clean_descriptions(descriptions)
vocabulary = to_vocabulary(descriptions)
print("Vocabulary size: ", len(vocabulary))

save_vocabulary(vocabulary, "vocabulary.txt")
save_descriptions(descriptions, "descriptions.txt")
