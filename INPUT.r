# Input:
inp <- head(df, 1)
inp$bat_team_Kolkata_Knight_Riders = 0
inp$bowl_team_Royal_Challengers_Bangalore = 0
inp$runs = 90
inp$wickets = 2
inp$overs = 10.0
inp$runs_last_5 = 50
inp$wickets_last_5 = 1
bat_team = "Chennai_Super_Kings"
bowl_team = "Mumbai_Indians"
if(bat_team == "Chennai_Super_Kings") inp$bat_team_Chennai_Super_Kings = 1 
if(bat_team == "Delhi_Daredevils") inp$bat_team_Delhi_Daredevils = 1
if(bat_team == "Kings_XI_Punjab") inp$bat_team_Kings_XI_Punjab = 1
if(bat_team == "Kolkata_Knight_Riders") inp$bat_team_Kolkata_Knight_Riders = 
1
if(bat_team == "Mumbai_Indians") inp$bat_team_Mumbai_Indians = 1
if(bat_team == "Rajasthan_Royals") inp$bat_team_Rajasthan_Royals = 1
if(bat_team=="Royal_Challengers_Bangalore")
inp$bat_team_Royal_Challengers_Bangalore = 1
if(bat_team == "Sunrisers_Hyderabad") inp$bat_team_Sunrisers_Hyderabad = 
1
if(bowl_team == "Chennai_Super_Kings") inp$bowl_team_Chennai_Super_Kings 
= 1
if(bowl_team == "Delhi_Daredevils") inp$bowl_team_Delhi_Daredevils = 1
if(bowl_team == "Kings_XI_Punjab") inp$bowl_team_Kings_XI_Punjab = 1
if(bowl_team == "Kolkata_Knight_Riders") inp$bowl_team_Kolkata_Knight_Riders 
= 1
if(bowl_team == "Mumbai_Indians") inp$bowl_team_Mumbai_Indians = 1
if(bowl_team == "Rajasthan_Royals") inp$bowl_team_Rajasthan_Royals = 1
if(bowl_team == "Royal_Challengers_Bangalore") 
inp$bowl_team_Royal_Challengers_Bangalore = 1
if(bowl_team == "Sunrisers_Hyderabad") inp$bowl_team_Sunrisers_Hyderabad 
= 1
#Predict for above input
pred <- predict(slr, newdata = inp)
print(paste("Predicted Score: ", round(pred, 0)))
B) IPL MATCH WINNER CODE:
rm(list = ls())
setwd("C:/6th SEM MATERIALS/Essentials of Data Analytics -Theory/J 
Component/Sanjay Files/Sanjay Files")
library(dplyr)
# loading dataset using read_csv()
df = read.csv('pure_data.csv')
### bring DV to first col
df <- df %>%select("team1_win", everything())
# displaying data
head(df)
str(df)
df$team1 = as.factor(df$team1)
df$team2 = as.factor(df$team2)
df$team1_toss_win = as.factor(df$team1_toss_win)
df$team1_win = as.factor(df$team1_win)
df$venue = as.factor(df$venue)
str(df)
# Train test split
## 85% of the sample size
smp_size <- floor(0.85 * nrow(df))

## set the seed to make your partition reproducible
set.seed(321)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
train <- df[train_ind, ]
test <- df[-train_ind, ]
dim(train)
dim(test)
# Creating formula
independentvariables=colnames(df[,2:5])
independentvariables
Model=paste(independentvariables,collapse="+")
Model
Model=paste("team1_win~",Model)
formula=as.formula(Model)
formula
# Logistic Regression
model=glm(formula=formula,data=train,family="binomial")
model=step(object = model,direction = "both")
pred=ifelse(test=model$fitted.values>0.5,yes = 1,no=0)
CFM1 <- table(model$y,pred)
accuracy1 <- sum(diag(CFM1)) / sum(CFM1)

accuracy1
# SVM
# install.packages('e1071')
library(e1071)
classifier = svm(formula = formula,data = train, type = 'C-classification', kernel = 'linear')
pred = predict(classifier, newdata = train)
CFM2 = table(train$team1_win, pred)
CFM2
accuracy2 <- sum(diag(CFM2)) / sum(CFM2)
accuracy2
# XGBOOST
#install.packages("xgboost")
library(xgboost)
X_train = data.matrix(df[,-1]) # independent variables for train
y_train = df[,1] # dependent variables for train
# convert the train and test data into xgboost matrix type.
xgboost_train = xgb.DMatrix(data=X_train, label=y_train)
# train a model using our training data
model <- xgboost(data = xgboost_train, # the data 
 max.depth=3, , # max depth 
 nrounds=50) # max number of boosting iterations


summary(model)
#use model to make predictions on test data
pred = predict(model, xgboost_train)
pred=ifelse(test=pred >= 1.5,yes=1,no=0)
CFM3 <- table(y_train,pred)
accuracy3 <- sum(diag(CFM3)) / sum(CFM3)
accuracy3
print(paste("LOG: ", accuracy1))
print(paste("SVM: ", accuracy2))
print(paste("XGB: ", accuracy3))
##TESTING
df_pred = read.csv('test_winner_prediction.csv')
### bring DV to first col
df_pred <- df_pred %>%select("team1_win", everything())
# displaying data
head(df_pred)
str(df_pred)
df_pred$team1 = as.factor(df_pred$team1)
df_pred$team2 = as.factor(df_pred$team2)
df_pred$team1_toss_win = as.factor(df_pred$team1_toss_win)
df_pred$team1_win = as.factor(df_pred$team1_win)

df_pred$venue = as.factor(df_pred$venue)
str(df_pred)
# prediction with xgboost
X_test = data.matrix(df_pred[,-1]) # independent variables for train
y_test = df_pred[,1] # dependent variables for train
# convert the train and test data into xgboost matrix type.
xgboost_test = xgb.DMatrix(data=X_test, label=y_test)
#use model to make predictions on test data
pred = predict(model, xgboost_test)
pred=ifelse(test=pred >= 1.5,yes=1,no=0)
CFM4 <- table(y_test,pred)
accuracy4 <- sum(diag(CFM4)) / sum(CFM4)
accuracy4
print(paste("Test: ", accuracy4))
