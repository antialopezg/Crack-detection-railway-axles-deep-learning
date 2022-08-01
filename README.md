# Railway axle crack diagnosis
by Antía López Galdo, Alejandro Guerrero-López, Pablo M. Olmos and María Jesús Gómez García

This paper has been submitted for publication in Reliability Engineering and System Safety.

We propose a novel method to classify vibration signals from accelerometers of railway axles. To do so, we propose to transform the vibration signals into spectrograms and work in the time-frequency domain, obtaining 3-layered images:
![speccc](https://user-images.githubusercontent.com/79870718/182141897-51532c80-e556-4e41-ab1e-eeb0caeab74f.png)

The method includes a two-dimensional convolutional neural network combined with a multilayer perceptron to classify the spectrograms in 4 different classes. The model architecture is displayed in the following figure:
![NN_3 (1)](https://user-images.githubusercontent.com/79870718/182142485-8875126d-b221-4545-9a80-40efd3ef1958.png)


# Abstract 
Railway axle maintenance is key to avoid catastrophic failures. Nowadays, condition monitoring techniques are becoming more prominent in the industry to prevent from huge costs and damages to human lives. 

In this work, signal processing techniques and Deep Learning models are proposed for effective railway axle crack detection based on vibration signals from accelerometers. To do so, several preprocessing steps, and different types of Neural Network architectures are discussed to design an accurate classification system. The resultant system converts the railway axle vibration signals into time-frequency domain representations, i.e. spectrograms, and, thus, train a two-dimensional convolutional neural networks to classify them depending on their cracks. This 2D-NN architecture, based on spectrograms, has been tested over 4 datasets achieving outperforming 0.93, 0.86, 0.75, 0.93 AUC scores showing a high level of reliability when classifying 4 different levels of defect.

# Software implementation

All source code used to generate the results and figures in the paper are in the following files:

- 2DCNN_ref: contains the code about the main experiment in the paper.

- 2DCNN: contains the code about the first alternative experiment in the paper.

- 1DCNN_LSTM: contains the code about the second alternative experiment in the paper.

- mlmodel: contains the code about the third alternative experiment in the paper.

The data used in this study is publicly available. You should create a folder called data and download WS and F datasets. Due to their size, we do not upload them to this repo.
