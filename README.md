# Detecting train driveshaft damages using accelerometer signals and Differential Convolutional Neural Networks
by Antía López Galdo, Alejandro Guerrero-López, Pablo M. Olmos and María Jesús Gómez García

This paper has been submitted for publication in Reliability Engineering and System Safety.

We propose a novel method to classify vibration signals from accelerometers of railway axles. To do so, we propose to transform the vibration signals into spectrograms and work in the time-frequency domain, obtaining 3-layered images:
![speccc](https://user-images.githubusercontent.com/79870718/201116248-c2b36785-dc34-4275-bfac-9cfef760b1ac.png)

The method includes a two-dimensional differential convolutional neural network combined with a multilayer perceptron to classify the spectrograms in 4 different classes. The model architecture is displayed in the following figure:
![Untitled 002](https://user-images.githubusercontent.com/79870718/201115910-d239b215-75ec-4589-b3c4-04b8b7c17f0c.jpeg)


## Abstract 
Railway axle maintenance is critical to avoid catastrophic failures. Nowadays,
condition monitoring techniques are becoming more prominent in the industry
to prevent enormous costs and damage to human lives.
This paper proposes the development of a railway axle condition monitoring
system based on advanced 2D-Convolutional Neural Network (CNN) architectures applied to time-frequency representations of vibration signals. For this
purpose, several preprocessing steps and different types of Deep Learning (DL)
and Machine Learning (ML) architectures are discussed to design an accurate
classification system. The resultant system converts the railway axle vibration signals into time-frequency domain representations, i.e., spectrograms, and,
thus, trains a two-dimensional CNN to classify them depending on their cracks.
The results showed that the proposed approach outperforms several alternative
methods tested. The CNN architecture has been tested in 3 different wheelset
assemblies, achieving AUC scores of 0.93, 0.86, and 0.75 outperforming any other
architecture and showing a high level of reliability when classifying 4 different
levels of defects.


## Software implementation

All source code used to generate the results and figures in the paper are in the following files:

- 2DCNN_ref: contains the code about the main experiment in the paper.

- 2DCNN: contains the code about the first alternative experiment in the paper.

- 1DCNN_LSTM: contains the code about the second alternative experiment in the paper.

- mlmodel: contains the code about the third alternative experiment in the paper.

The data used in this study is publicly available. You should create a folder called data and download WS and F datasets. Due to their size, we do not upload them to this repo.
