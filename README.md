# Medical Text Classification using Dissimilarity Space

![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/Medical_Transcription.jpg?raw=true)

## Goal

This project goal is to build a classifier which classify medical specialties based on the thier transcription text.

## Motivation

## Main Idea
The main idea used in this project is to learn a smart embedding to each medical transcription in the training set, and then use the embedded vectors to train classifiers. Then one can perform the same embedding to a new medical transcription and predict its 

## Resources
[1] Spectrogram Classification Using Dissimilarity Space: https://www.mdpi.com/2076-3417/10/12/4176/htm

## Libaries
Pytorch, HuggingFace, sklearn,  Numpy, Plotly

## Data
Data contains 4966 rows, each including three main elements: <br/>
**transcription** - Medical transcription of some patient (text).  <br/>
**description** - Short description of transcription (text).  <br/>
**medical_specialty** - Medical specialty classification of transcription (category). There are 40 different categories. Figure 2 displays the distribution of the categories in the dataset.


| ![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/categories_dists.png?raw=true)|
| Figure 1: Categories Distribution |

Due to limitations in time and memory, we use descriptions rather than transcriptions (see Figure 3 and Figure 4, which displays their text length histograms)


| ![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/text_length.png?raw=true) | ![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/text_length_transcription.png?raw=true) |
| --- | --- |
| Figure 2:  text length discription| Figure 3: text length transcription|

## Why using Dissimilarity Sapce and not Direct classifier

- imbalanced data
- new categories: no need to train new model?
- To try this method, self learing....

## Scheme
| ![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/Scheme.png?raw=true) |
| --- |
| Training Scheme |


## Sieamese Neural Network

![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/giraffes.jpg?raw=true)

## Results

![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/model_loss.png?raw=true)

