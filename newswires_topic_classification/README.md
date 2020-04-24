# Reuters Newswires Topic Classification 

## Summary
Used Keras and Tensorflow to predict the topics of newswires based on their text content.        
* Trained a neural network to classify over 10k Reuters newswires into 46 topics. 
* Fine-tuned the network and obtained an accuracy of 78%. 
* Performed different parameters like the learning rate, batch size, and gradient approach. Improved the accuracy by 2% on the test set.

# Packages
* **Tensorflow:** version 1.14
* **Keras:** version 2.3.1
* numpy: 1.16.1

## The Reuters dataset
*  The _Reuters dataset_, a set of short newswires and their topics, published by Reuters in 1986. 
* We have 8,982 training examples and 2,246 test examples

# Classifying newswires: a multi-class classification
* Goal: to classify Reuters newswires into 46 different mutually-exclusive topics. 
* This problem is an instance of "multi-class classification", and since each data point should be classified into only one category, the problem is more specifically an instance of **"single-label, multi-class classification"**.

# Neural Network Model
* Sequential model
* Ending the network with a `Dense` layer of size 46. 
* The last layer uses a `softmax` activation. 
* output: output a _probability distribution_ over the 46 different output classes, i.e. for every input sample, the network will produce a 46-dimensional output vector where `output[i]` is the probability that the sample belongs to class `i`. The 46 scores will sum to 1.
* loss function is `categorical_crossentropy`. It measures the distance between two probability distributions: in this case, between the probability distribution output by our network, and the true distribution of the labels. By minimizing the distance between these two distributions, we train our network to output something as close as possible to the true labels.