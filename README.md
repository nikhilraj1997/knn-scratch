# K-Nearest Neighbours (KNN) From Scratch

This repository is an implmentation of K-Nearest Neighbours (KNN) from scratch by just using NumPy as it's main processing library. It also includes k-fold cross-validation that has been implemented from scratch. Dataset used here is a binary image dataset for handwritten numbers. It is similar to [MNIST dataset](http://yann.lecun.com/exdb/mnist/) but it is not exactly the same.

## Running the program

The code resides in the file `kNN.py`. To run it simply run it as a usual Python program.

## Changable Parameters

There are two parameters:

1. `NO_OF_FOLDS` defines the number of folds in k-fold cross validation
2. `NO_OF_NEIGHBORS` defines the maximum number of neighbours to be considered 

## Methodology

For a given number of `NO_OF_NEIGHBORS` the given dataset is iterated through 1 to `NO_OF_NEIGHBORS` only considering odd numbers. For each iteration k-fold cross validation is also performed. The distance metric is Euclidean distance. Other distancs were not included to reduce the overall time taken for the program to run.

## Accuracy and output

Although done for a single dataset, the accuracy is around 99% for training. Please feel free to try with other datasets (you will need to cdit some helper functions).

A typical output is as follows:

```
==================================================
Current fold: 1
________________________________________
Training for fold: 1
k is 3
Classification errors: 20, Training set accuracy: 98.70801033591732
k is 5
Classification errors: 29, Training set accuracy: 98.1266149870801
k is 7
Classification errors: 34, Training set accuracy: 97.80361757105943
k is 9
Classification errors: 40, Training set accuracy: 97.41602067183463
k is 11
Classification errors: 41, Training set accuracy: 97.3514211886305
________________________________________
Validating for fold: 1
Classification errors: 8 Validation set accuracy: 0.9792746113989638
==================================================
Current fold: 2
________________________________________
(1162, 1024)
Training for fold: 2
k is 3
Classification errors: 13, Training set accuracy: 98.88123924268503
k is 5
Classification errors: 23, Training set accuracy: 98.02065404475043
k is 7
Classification errors: 27, Training set accuracy: 97.67641996557659
k is 9
Classification errors: 32, Training set accuracy: 97.24612736660929
k is 11
Classification errors: 35, Training set accuracy: 96.98795180722891
________________________________________
Validating for fold: 2
Classification errors: 7 Validation set accuracy: 0.9818652849740933
==================================================
Current fold: 3
________________________________________
(1162, 1024)
Training for fold: 3
k is 3
Classification errors: 15, Training set accuracy: 98.7091222030981
k is 5
Classification errors: 24, Training set accuracy: 97.93459552495698
k is 7
Classification errors: 29, Training set accuracy: 97.50430292598968
k is 9
Classification errors: 32, Training set accuracy: 97.24612736660929
k is 11
Classification errors: 34, Training set accuracy: 97.07401032702238
________________________________________
Validating for fold: 3
Classification errors: 9 Validation set accuracy: 0.9766839378238342
==================================================
Current fold: 4
________________________________________
(1162, 1024)
Training for fold: 4
k is 3
Classification errors: 13, Training set accuracy: 98.88123924268503
k is 5
Classification errors: 23, Training set accuracy: 98.02065404475043
k is 7
Classification errors: 29, Training set accuracy: 97.50430292598968
k is 9
Classification errors: 33, Training set accuracy: 97.16006884681583
k is 11
Classification errors: 39, Training set accuracy: 96.64371772805508
________________________________________
Validating for fold: 4
Classification errors: 9 Validation set accuracy: 0.9766839378238342
==================================================
Current fold: 5
________________________________________
Training for fold: 5
k is 3
Classification errors: 19, Training set accuracy: 98.76943005181347
k is 5
Classification errors: 32, Training set accuracy: 97.92746113989638
k is 7
Classification errors: 37, Training set accuracy: 97.60362694300518
k is 9
Classification errors: 41, Training set accuracy: 97.34455958549223
k is 11
Classification errors: 44, Training set accuracy: 97.15025906735751
________________________________________
Validating for fold: 5
Classification errors: 12 Validation set accuracy: 0.9692307692307692
==================================================
Testing for given test dataset
________________________________________
Classification errors for testing dataset: 8
Testing accuracy with the the best yet k, k=3 is 99.15433403805497%
```

## Reviews and feedback

I am not clainming that this is the best from scracth solution available for KNN by any means so any feedback, reviews and comments are welcome.