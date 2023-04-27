# Google - Isolated Sign Language Recognition Kaggle Competition
This is a repo containing the code I worked on with Jackson Bolos for this Kaggle Competition:
https://www.kaggle.com/competitions/asl-signs/data

###Introduction
From Kaggle:
The goal of this competition is to classify (250) isolated American Sign Language (ASL) signs. You will create a TensorFlow Lite model trained on labeled landmark data extracted using the MediaPipe Holistic Solution.

Your work may improve the ability of PopSign to help relatives of deaf children learn basic signs and communicate better with their loved ones.

Vocab & Notes:
- Isolated means the signs are one sign at a time, not a 'sentence' of signs
- TensorFlow lite models are being used here so they can run on phones and edge devices
- MediaPipe is a program which takes images and videos and recognizes 'landmarks' such as faces and hands. It then tracks the x, y and z movements of these landmarks over time. This makes computation here more efficient as we don't have to process raw video, instead just sequences of landmark movement.
- PopSign is the intended user of the model, which is an where they're hoping to enable a feature that prompts the user with five words (bubbles in the app's game) they can 'pop' by signing. 

### Getting Started
To get started with this repository, follow these steps:

Clone the repository
Install the required packages by running pip install -r requirements.txt
Download the dataset from the Kaggle competition page
Load the data into the appropriate format using Pandas or NumPy
Train a machine learning model using Scikit-Learn, TensorFlow, or PyTorch
Use the trained model to make predictions on the test dataset
Submit the predictions to the Kaggle competition page

### Repository Structure
This repository contains the following files:
.gitignore: things for git to not track the changes on
requirements.txt: a list of required Python packages
README.md: a description of the repository
KaggleCompWriteUp.docx: A Brief summarization of our experience in the challenge and what we learned
TraditionalModels.ipynb: an attempt to use machine learning models toward the challenge as opposed to neural networks
ModelSubmission: our EDA work and first submission to the competition
images: screenshots required for the notebooks
models: converted tflite models
OtherHelpfulKaggleNotebooks: notebooks that greatly assisted in our attempts on this challenge
