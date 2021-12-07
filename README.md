# Medical Text Classification using Dissimilarity Space

![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/Medical_Transcription.jpg?raw=true)

## Goal

This project goal is to build a classifier which classify medical specialties based on the thier transcription text.

## Motivation

## Main Idea
The main idea used in this project is to learn a smart embedding to each medical transcription in the training set, and then use the embedded vectors to train classifiers. Then one can perform the same embedding to a new medical transcription and predict its...

The idea is adapted from [1], with the necessary adjustments because text data is used in this project instead of images data.

## Resources
[1] Spectrogram Classification Using Dissimilarity Space: https://www.mdpi.com/2076-3417/10/12/4176/htm

## Libaries
Pytorch, HuggingFace, sklearn,  Numpy, Plotly

## Data
The original data contains 4966 records, each including three main elements: <br/>

**transcription** - Medical transcription of some patient (text).  <br/>

**description** - Short description of the transcription (text).  <br/>

**medical_specialty** - Medical specialty classification of transcription (category).  <br/>
There are 40 different categories. Figure 1 displays the distribution of the top 20 categories in the dataset.

<br/>

| ![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/categories_dists.png?raw=true)|
| --- |
| **Figure 1**: Top 20 categories|


One can see that the dataset is very unbalanced - most categories represent less than 5% of the total,  each. 

Due to limitations in time and memory, I use descriptions rather than transcriptions (see Figure 3 and Figure 4, which displays their text length histograms).


| ![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/descriptions_length.png?raw=true) | ![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/transcriptions_length.png?raw=true) |
| --- | --- |
| **Figure 2**: Description texts length| **Figure 3**: Transcription texts length|

## Why using Dissimilarity Sapce and not Direct classifier

- imbalanced data
- new categories: no need to train new model?
- To try this method, self learing....

## Scheme
The training procedure consists of several steps which are schematized in Figure 4.

| ![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/Scheme.png?raw=true) |
| --- |
| **Figure 4**: Training Scheme |

**Siamese Neural Network (SNN) Training** <br/>
The purpose of this phase is to learn a distance measure d(x,y) by maximizing the similarity between couples of samples in the same category, while minimizing the similarity for couples in different categories. <br/>
Our siamese network model consists of several components:

- **Two identical twin subnetworks** <br/>
two identical sub-network that share the same parameters and weights. Each subnetwork gets as input a text and outputs a feature vector which is designed to represent the text. I chose as a subnetwork a pre-trained Bert model (a huggingface model which trained on abstracts from PubMed, see [2]) followed by a FF layer for fine-tuning.
- **Subtract Block** <br/>
Subtracting the output feature vectors of the subnetworks yields a feature vector Y representing the difference between the texts: Y = | f1 - f2 | <br/>
- **Fully Connected Layer** (FCL) <br/>
Learn the distance model to calculate the dissimilarity. The output vector of the subtract block is fed to the FCL which returns a dissimilarity value for the pair of texts in the input.Then  a sigmoid function is applied  to the dissimilarity value to convert it to a probability value in the range [0, 1].


Binary Cross Entropy


**Prototype Selection** <br/>
In this phase, K prototypes are extracted from the training set. As the autores of [1] stated, it is not practical to take every sample in the training as a prototype. Alternatively, m centroids for each category separately are computed by clustering technique. This reduces the prototype list from the size of the training sample (K=n) to K=m*C (C=number of categories). I chose K-means for the clustering algorithm.

In order to represent the training samples as vectors for the clustering algorithm, the authors in [1] used the pixel vector of each image. In this project, I utilize the embedding layers of the trained SNN to retrieve the feature vectors of every training sample.

**Projection in the Dissimilarity Space** <br/>
In this phase the data is projected into dissimilarity space. In order to obtain the representation of a sample in the dissimilarity space,we calculate the  similarity between the sample and the selected set of prototypes P=p1,...pk, which resulting in a dissimilarity vector:
F(x)=[d(x,pi),d(x,pi+1),...,d(x,pk)],
The similarity among a sample and a prototype d(x,y) is obtained using the trained SNN.

**SVM Classifiers** <br/>
In this phase an ensemble of SVMs are trained using a One-Against-All approach: For each category an SVM is trained to discriminate between this category and all the other categories put together. The sample is then assigned to the category that gives the highest confidence score. The inputs for the classifiers are the projected train data.


## Sieamese Neural Network

![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/giraffes.jpg?raw=true)

## Results

![pic](https://github.com/OdedMous/Medical-Transcriptions-Classification/blob/main/images/model_loss.png?raw=true)

