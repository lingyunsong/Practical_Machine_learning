Practical Machine Learning: Prediction Assignment Writeup
=========================================================

Project requirement:
--------------------
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their BEHAVIOR, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and INCORRECTLY in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 
The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the DOCUMENT you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

What you should SUBMIT

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross VALIDATION, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you SUBMIT a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the PROGRAMMING assignment for automated grading. See the programming assignment for additional details. 

Background of the data
----------------------
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3XFG6Fcoy

Data Loading
------------
#set up directory:

```r
setwd("~/Desktop/Coursera/Practical_Machine_learning")
```
#download and read training data and test data:

```r
url1="http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2="http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url1, destfile = "train.csv", method="auto")
download.file(url2, destfile = "test.csv", method="auto")
train_data <- read.csv("train.csv", sep=',', header=T)
testing <- read.csv("test.csv", sep=',', header=T)
```
Data Preprocessing
------------------
#check missing values in train.csv file, by rows or by columns

```r
#how many rows (observations) have missing data: 
nrow(train_data[!complete.cases(train_data),])
```

```
## [1] 19216
```

```r
#How many columns (variables) have missing data:
col_with_miss <- which(apply(train_data, 2, function(x) sum(x %in% c(NA, '')))> 0)
length(col_with_miss) 
```

```
## [1] 100
```
#For data cleaning there are two options to treat missing values: Either delete missing values by rows or by columns. More than 97% of observations will be gone if I get rid of all rows containing missing data; and ca. 62% of variables will be gone if I remove variables containing missing data. So I chose to delete variables containing "NA" or nothing. After that, the non-numberic variables (from column 1 to column 6) are removed to meet the model fitting requirement. After data cleaning, new training dataset keeps the same observations with original training data but has 54 variables instead of 160.

```r
train_data <- train_data[,-col_with_miss]
train_data <- train_data[,-c(1:6)]
dim(train_data)
```

```
## [1] 19622    54
```
Machine Learning and cross-validation
-------------------------------------
#In the following steps, original training data are split into two subsets: training (80%) and validation (20%). The training subset is fitted with three commonly used methods:decision trees(rpart), random forest(rf), and boosted trees(gbm), respectively. 

```r
#split data to training part and validation part:
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.2
```

```r
inTrain <- createDataPartition(y=train_data$classe, p=0.8, list=F)
set.seed(1234)
validation <- train_data[-inTrain,]
training <- train_data[inTrain,]
dim(training)
```

```
## [1] 15699    54
```

```r
dim(validation)
```

```
## [1] 3923   54
```

```r
#fit training subset with three machine learning algorithms:
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(rpart)
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.1.3
```

```
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1.1
```

```r
library(plyr)
set.seed(5678)
rfmodFit <- train(classe~ ., data=training, method="rf", trControl = trainControl(method="cv"),number=3)
rpmodFit <- train(classe~ ., data=training, method="rpart")
boostFit <- train(classe~ ., method="gbm", data=training, verbose=F)

#Predict validation classe by using 3 fitted models:
pred_rf_v <- predict(rfmodFit, validation)
pred_rp_v <- predict(rpmodFit, validation)
pred_boostFit_v <- predict(boostFit, validation)
```

In-sample error and out-of-sample error analysis
------------------------------------------------

```r
#out-of-sample errors
confusionMatrix(pred_rf_v, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  758    0    0    0
##          C    0    0  684    4    0
##          D    0    0    0  639    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9987         
##                  95% CI : (0.997, 0.9996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9984         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9987   1.0000   0.9938   1.0000
## Specificity            0.9996   1.0000   0.9988   1.0000   1.0000
## Pos Pred Value         0.9991   1.0000   0.9942   1.0000   1.0000
## Neg Pred Value         1.0000   0.9997   1.0000   0.9988   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1932   0.1744   0.1629   0.1838
## Detection Prevalence   0.2847   0.1932   0.1754   0.1629   0.1838
## Balanced Accuracy      0.9998   0.9993   0.9994   0.9969   1.0000
```

```r
confusionMatrix(pred_rp_v, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1015  334  306  294  102
##          B   21  225   27  106   98
##          C   77  200  351  243  198
##          D    0    0    0    0    0
##          E    3    0    0    0  323
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4879          
##                  95% CI : (0.4721, 0.5037)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3305          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9095  0.29644  0.51316   0.0000  0.44799
## Specificity            0.6309  0.92035  0.77833   1.0000  0.99906
## Pos Pred Value         0.4949  0.47170  0.32834      NaN  0.99080
## Neg Pred Value         0.9460  0.84504  0.88332   0.8361  0.88935
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2587  0.05735  0.08947   0.0000  0.08233
## Detection Prevalence   0.5228  0.12159  0.27250   0.0000  0.08310
## Balanced Accuracy      0.7702  0.60840  0.64574   0.5000  0.72353
```

```r
confusionMatrix(pred_boostFit_v, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1113    8    0    0    0
##          B    3  744    5    3    5
##          C    0    6  678    6    1
##          D    0    1    1  634    4
##          E    0    0    0    0  711
## 
## Overall Statistics
##                                           
##                Accuracy : 0.989           
##                  95% CI : (0.9853, 0.9921)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9861          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9973   0.9802   0.9912   0.9860   0.9861
## Specificity            0.9971   0.9949   0.9960   0.9982   1.0000
## Pos Pred Value         0.9929   0.9789   0.9812   0.9906   1.0000
## Neg Pred Value         0.9989   0.9953   0.9981   0.9973   0.9969
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2837   0.1897   0.1728   0.1616   0.1812
## Detection Prevalence   0.2858   0.1937   0.1761   0.1631   0.1812
## Balanced Accuracy      0.9972   0.9876   0.9936   0.9921   0.9931
```

```r
#in-sample error:
pred_rf_t <- predict(rfmodFit, training)
pred_rp_t <- predict(rpmodFit, training)
pred_boostFit_t <- predict(boostFit, training)
confusionMatrix(pred_rf_t, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
confusionMatrix(pred_rp_t, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4061 1241 1280 1153  421
##          B   66 1068   82  466  392
##          C  326  729 1376  954  765
##          D    0    0    0    0    0
##          E   11    0    0    0 1308
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4977          
##                  95% CI : (0.4898, 0.5055)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3435          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9097  0.35155  0.50256   0.0000  0.45322
## Specificity            0.6355  0.92054  0.78597   1.0000  0.99914
## Pos Pred Value         0.4979  0.51495  0.33157      NaN  0.99166
## Neg Pred Value         0.9466  0.85541  0.88207   0.8361  0.89026
## Prevalence             0.2843  0.19352  0.17441   0.1639  0.18383
## Detection Rate         0.2587  0.06803  0.08765   0.0000  0.08332
## Detection Prevalence   0.5195  0.13211  0.26435   0.0000  0.08402
## Balanced Accuracy      0.7726  0.63605  0.64426   0.5000  0.72618
```

```r
confusionMatrix(pred_boostFit_t, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4461   10    0    0    0
##          B    2 3007   13    8    4
##          C    0   17 2718   23    4
##          D    1    4    6 2541   14
##          E    0    0    1    1 2864
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9931          
##                  95% CI : (0.9917, 0.9944)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9913          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9898   0.9927   0.9876   0.9924
## Specificity            0.9991   0.9979   0.9966   0.9981   0.9998
## Pos Pred Value         0.9978   0.9911   0.9841   0.9903   0.9993
## Neg Pred Value         0.9997   0.9976   0.9985   0.9976   0.9983
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1915   0.1731   0.1619   0.1824
## Detection Prevalence   0.2848   0.1933   0.1759   0.1634   0.1826
## Balanced Accuracy      0.9992   0.9938   0.9947   0.9928   0.9961
```
#above results show that random forest is the most accurate method for obtaining smallest in-sample errors and out-of-sample errors.Boosting is a little less accurate than rf, but still gets accuracies at ca. 99%. The method of decision trees is much less inaccurate than the other two.
Since the overall accuracy of random forest prediction at validation dataset is 0.9987, and in-sample accuracy is 1. This suggested out-of-sample error(0.13%) is slightly larger than in-sample error, and similar error value should be observed in predicting testing dataset.

Predict observations in testing dataset by random Forest
--------------------------------------------------------

```r
testing <- testing[,-col_with_miss]
testing <- testing[,-c(1:6)]
dim(testing)
```

```
## [1] 20 54
```

```r
predict(rfmodFit, testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


