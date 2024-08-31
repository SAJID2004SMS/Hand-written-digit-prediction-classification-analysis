# Hand-written-digit-prediction-classification-analysis
DESCRIPTION 
The digits dataset consists of 8x8 pixel images pf digits.The images attribute of the dataset stores 8x8 arrays of grayscale values for each image.we will use these arrays to visualize the first 4 images.The target attritube of the dataset stores the digit each image represents

ybi foundation-github

Import Library

[1]
1s
import pandas as pd

[2]
0s
import numpy as np

[3]
0s
import matplotlib.pyplot as plt
Import Data

[4]
3s
from sklearn.datasets import load_digits
Describe Data

[5]
0s
df = load_digits()
Data Visualization

[6]
0s
_, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (10, 3))
for ax, image, label in zip(axes, df.images, df.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation = 'nearest')
    ax.set_title('Training: %i' % label)

Data Preprocessing
Flatten Image

[7]
0s
df.images.shape
(1797, 8, 8)

[8]
1s
df.images[0]
array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],
       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],
       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],
       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],
       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],
       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],
       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],
       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])

[9]
0s
df.images[0].shape
(8, 8)

[10]
0s
n_samples = len(df.images)
data = df.images.reshape((n_samples, -1))

[11]
0s
data[0]

array([ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
       15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
       12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
        0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
       10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.])

[12]
0s
data[0].shape
(64,)

[13]
0s
data.shape
(1797, 64)
Scaling Data

[14]
0s
data.min()
0.0

[15]
0s
data.max()
16.0

[16]
0s
data = data/16

[17]
0s
data.min()
0.0

[18]
0s
data.max()
1.0

[19]
0s
data[0]
array([0.    , 0.    , 0.3125, 0.8125, 0.5625, 0.0625, 0.    , 0.    ,
       0.    , 0.    , 0.8125, 0.9375, 0.625 , 0.9375, 0.3125, 0.    ,
       0.    , 0.1875, 0.9375, 0.125 , 0.    , 0.6875, 0.5   , 0.    ,
       0.    , 0.25  , 0.75  , 0.    , 0.    , 0.5   , 0.5   , 0.    ,
       0.    , 0.3125, 0.5   , 0.    , 0.    , 0.5625, 0.5   , 0.    ,
       0.    , 0.25  , 0.6875, 0.    , 0.0625, 0.75  , 0.4375, 0.    ,
       0.    , 0.125 , 0.875 , 0.3125, 0.625 , 0.75  , 0.    , 0.    ,
       0.    , 0.    , 0.375 , 0.8125, 0.625 , 0.    , 0.    , 0.    ])
Train test split Data

[20]
0s
from sklearn.model_selection import train_test_split

[21]
0s
xtrain, xtest, ytrain, ytest = train_test_split(data, df.target, test_size = 0.3)

[22]
0s
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape
((1257, 64), (540, 64), (1257,), (540,))
Modeling
Random Forest model

[23]
1s
from sklearn.ensemble import RandomForestClassifier

[24]
0s
rf = RandomForestClassifier()
Train or Fit Model

[25]
1s
rf.fit(xtrain, ytrain)

Prediction
Predict Test Data

[26]
0s
y_pred = rf.predict(xtest)

[27]
0s
y_pred
array([0, 8, 3, 8, 6, 3, 9, 1, 5, 2, 4, 0, 7, 1, 2, 0, 3, 4, 0, 8, 6, 2,
       2, 4, 7, 1, 1, 4, 8, 3, 0, 5, 1, 0, 3, 7, 0, 9, 8, 4, 9, 7, 8, 3,
       9, 4, 1, 3, 5, 1, 1, 9, 8, 6, 8, 0, 3, 9, 1, 8, 5, 1, 2, 2, 9, 2,
       7, 9, 4, 9, 5, 2, 7, 6, 2, 8, 0, 9, 7, 4, 2, 1, 2, 6, 4, 5, 0, 6,
       4, 7, 0, 0, 1, 2, 3, 6, 5, 4, 0, 1, 7, 8, 1, 9, 2, 4, 1, 4, 4, 3,
       8, 8, 4, 7, 8, 9, 0, 1, 8, 5, 5, 0, 4, 0, 2, 1, 5, 1, 3, 9, 0, 7,
       5, 2, 6, 5, 9, 4, 9, 0, 6, 7, 6, 4, 6, 4, 4, 5, 9, 2, 2, 4, 6, 0,
       7, 8, 6, 6, 4, 7, 2, 3, 8, 9, 5, 3, 1, 1, 0, 1, 0, 2, 4, 7, 4, 1,
       1, 3, 4, 3, 4, 0, 9, 2, 4, 4, 7, 9, 8, 4, 2, 5, 6, 7, 6, 0, 5, 7,
       2, 1, 8, 3, 0, 3, 1, 8, 4, 4, 1, 4, 0, 9, 6, 4, 7, 4, 9, 9, 9, 7,
       5, 3, 8, 9, 3, 4, 7, 6, 7, 8, 9, 9, 0, 3, 8, 5, 0, 2, 7, 3, 5, 3,
       4, 1, 2, 9, 7, 3, 0, 9, 4, 5, 5, 8, 8, 1, 0, 0, 6, 3, 1, 7, 1, 1,
       0, 9, 0, 8, 7, 9, 0, 9, 7, 1, 4, 5, 7, 8, 2, 4, 8, 2, 9, 3, 7, 2,
       1, 3, 2, 2, 3, 7, 9, 9, 3, 4, 0, 1, 7, 6, 2, 4, 9, 8, 5, 6, 3, 6,
       9, 5, 5, 5, 3, 7, 8, 4, 4, 3, 8, 4, 5, 0, 0, 5, 9, 6, 2, 3, 3, 0,
       3, 3, 3, 9, 1, 0, 5, 9, 6, 8, 2, 1, 9, 1, 1, 6, 6, 0, 6, 3, 1, 3,
       8, 6, 8, 4, 4, 6, 8, 0, 0, 7, 0, 0, 0, 6, 1, 9, 6, 7, 8, 8, 3, 7,
       0, 6, 3, 7, 1, 3, 5, 8, 8, 0, 0, 0, 6, 9, 2, 3, 9, 5, 7, 9, 6, 7,
       6, 5, 9, 4, 1, 4, 1, 1, 6, 8, 2, 5, 7, 8, 8, 1, 0, 2, 0, 5, 9, 8,
       6, 4, 9, 1, 9, 3, 2, 4, 4, 7, 1, 2, 6, 5, 9, 0, 1, 7, 3, 0, 1, 2,
       9, 1, 1, 9, 7, 8, 1, 9, 9, 5, 1, 9, 1, 0, 0, 3, 5, 4, 5, 7, 6, 0,
       3, 6, 4, 8, 2, 3, 6, 8, 5, 6, 5, 9, 3, 6, 8, 3, 6, 5, 1, 4, 9, 3,
       3, 7, 1, 3, 9, 1, 9, 0, 2, 4, 7, 7, 8, 6, 1, 9, 5, 0, 6, 4, 4, 6,
       7, 0, 9, 7, 6, 0, 6, 3, 6, 2, 2, 1, 3, 0, 9, 4, 1, 3, 1, 0, 4, 0,
       4, 5, 5, 9, 9, 2, 2, 7, 8, 2, 1, 2])
Model Evaluation
Model Accuracy

[28]
0s
from sklearn.metrics import confusion_matrix, classification_report

[29]
0s
confusion_matrix(ytest, y_pred)
array([[61,  0,  0,  0,  1,  0,  0,  0,  0,  0],
       [ 0, 60,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0, 46,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0, 55,  0,  1,  0,  0,  0,  0],
       [ 0,  0,  0,  0, 58,  0,  0,  2,  0,  0],
       [ 0,  0,  0,  0,  1, 40,  0,  0,  0,  2],
       [ 1,  0,  0,  0,  0,  0, 50,  0,  1,  0],
       [ 0,  0,  0,  0,  0,  0,  0, 47,  1,  0],
       [ 0,  2,  0,  0,  0,  1,  0,  0, 46,  0],
       [ 0,  0,  0,  0,  0,  2,  0,  0,  1, 61]])

[30]
0s
print(classification_report(ytest, y_pred))
              precision    recall  f1-score   support

           0       0.98      0.98      0.98        62
           1       0.97      1.00      0.98        60
           2       1.00      1.00      1.00        46
           3       1.00      0.98      0.99        56
           4       0.97      0.97      0.97        60
           5       0.91      0.93      0.92        43
           6       1.00      0.96      0.98        52
           7       0.96      0.98      0.97        48
           8       0.94      0.94      0.94        49
           9       0.97      0.95      0.96        64

    accuracy                           0.97       540
   macro avg       0.97      0.97      0.97       540
weighted avg       0.97      0.97      0.97       540

Explaination
Handwritten Digit Prediction, or Digit Classification, is like training a computer to read and understand handwritten numbers, similar to how we recognize them ourselves. Imagine you have a bunch of pictures, each showing a handwritten digit from 0 to 9. The goal is to teach the computer how to look at these pictures and figure out which digit is in each one. We split our pictures into two groups: one set to teach the computer (training set) and another to test its skills (testing set). We choose a smart method, like a special problem-solving recipe, for the computer to learn the patterns and differences in the pictures. As it learns, we guide it by telling whether its guesses are right or wrong. Once it masters this skill, we give it new pictures it hasn't seen during training, and it tries its best to guess the correct digit. We check its guesses to see how well it's doing, using simple checks to see if it's getting the numbers correct. This project helps us build a computer that can recognize handwritten digits, which has practical uses like reading postal codes or helping computers understand our handwriting. It's like teaching the computer to read your handwriting
