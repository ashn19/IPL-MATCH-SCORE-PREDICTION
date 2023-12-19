## Hey, I'm [Ashith Nirmal P!](https://github.com/ashn19) 👋.

### 📕 IPL MATCH SCORE PREDICTION
- IPL is a tournament to be noted among all the cricket fans all over the world. This paper focuses on two methods the first is prediction of IPL Match score and the second one is predicting the IPL Match Winner. These methods are implemented using most popular 
Machine Learning Algorithms and are simulated using R Studio Software.Two datasets are used in this base paper since two methods have to be propose through 
Machine learning techniques. Those two datasets are loaded and and a set of preprocessing is done along with the feature selection. 
In these two methods the first method focuses on Regression Problem and the second method focuses on Classification Problem. The Score Prediction include 3 Regression Models Linear 
Regression, Lasso Regression and Ridge Regression. The Match Winner Prediction includes 3 classification models Logistic Regression, SVM and XG Boost Classification model.These 
algorithms predicts with the help of functions of R Package.In the first method 3 algorithms are applied and the results are used to find and compare the accuracy and the mean squared error of those 3 models.In the second method, 3 algorithms are applied and are used to find the accuracy scores.The 
best model among them is then used for a testing dataset that predicts the IPL 2022 match winners in the league stage till date

### 📕 METHODOLOGY
# DATASET EXPLANATION:
Dataset for Score Prediction: For predicting the IPL Score we collected data of the IPL matches played from 2008-2017.The dataset contains 76014 rows and 15 columns.It undergoes 
set of preprocessing steps that reduces its columns from 15 to 8.The most important features are the runs in last five overs and the wickets in last 5 overs that is significant in our prediction.
We removed all the unwanted columns like mid,venue,batsman,bowler,striker,non striker and date since these variables are not required for our prediction.Dataset for Winner Prediction:For predicting the IPL match winner we collected data of the 
IPL matches played from 2017-2019.This dataset contains 756 rows and 18 columns.After preprocessing, the no of rows and columns are reduced to 754,5.Certain rows have been removed from the dataset since the match winner column had null values.
# DATA PREPROCESSING:
Removed redundancy in the teams by removing the duplicate teams and removing teams that participated in IPL for very few years.One hot Encoding is applied on the bat_team and bowl_team columns in the first method and 
label encoding is applied in the second method as we need to convert string to numerical categorical values.
# ALGORITHMS:
We have used regression algorithm for score prediction which include Simple Linear  Regression, Ridge Regression and Lasso Regression.
We have used classification algorithm for winner predcition which include Logistic regression, Support Vector Machines and XGBoost.
