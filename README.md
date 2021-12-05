# Medical Text Classification using Dissimilarity Space

![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/Medical_Transcription.jpg?raw=true)

## Goal
Medical Transcriptions Classification
This project goal is to build a classifier which classify medical specialties based on the thier transcription text.

## Motivation

## Main Idea
The main idea used in this project is to learn a smart embedding to each medical transcription in the training set, and then use the embedded vectors to train classifiers. Then one can perform the same embedding to a new medical transcription and predict its 

## Resources
Spectrogram Classification Using Dissimilarity Space: https://www.mdpi.com/2076-3417/10/12/4176/htm

## Libaries
Pytorch, HuggingFace, sklearn,  Numpy, Plotly

## Data
Data contains 4966 rows, each including three main columns:
transcription -Mmedical transcription
description - Short description of transcription
medical_specialty - Medical specialty classification of transcription



![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/text_length.png?raw=true)

## Why using Dissimilarity Sapce and not Direct classifier

- imbalanced data
- new categories: no need to train new model?
- To try this method, self learing....

## Scheme
![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/Scheme.png?raw=true)


## Sieamese Neural Network

![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/giraffes.jpg?raw=true)

## Results

![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/model_loss.png?raw=true)

