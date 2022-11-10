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

- 2D-CNN-Ref: contains the code implemented to develop the main experiment of the paper. This script is based on the design, training and testing of a two-dimensional differential convolutional neural network to classify spectrogram representations of the vibration series. This neural network is in turn applied to the spectrograms representing the healthy axle of the train, to provide the network with sufficient information to be able to differentiate standard system conditions from faulty ones.

- 2D-CNN (no reference): contains the code developed to assess the influence of using the reference data in the selected model. It consists of the two-dimensional differential neural network applied only to the spectrograms. 

- 1D-CNN-LSTM-Ref: this script provides the design of the one-dimensional convolutional neural network combined with the long-short-term-memory applied to the vibration signal to exploit the time dependency. This experiment allows us to analyze the value provided by the use of time-frequency representations of the data instead of using the raw signals in the time domain.

- tsfresh-RF: the tsfresh library was used to extract features from temporal signals and reduce dimensionality. In this script, these features were used as input to a random forest classifier to analyze the advantages of deep learning vs. classical machine learning techniques.

The data used in this study is not publicly available. 
