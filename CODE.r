A) IPL SCORE PREDICTION CODE:
rm(list = ls())
setwd("C:/6th SEM MATERIALS/Essentials of Data Analytics -Theory/J 
Component/Sanjay Files/Sanjay Files")
# loading dataset using read_csv()
df = read.csv('ipl - mod.csv')
# displaying data
head(df)
#checking shape of data using df.shape
dim(df)
str(df)
# Check for null
sum(is.na(df))
summary(df)
library(dplyr)
### bring DV to first col
df <- df %>%select("total", everything())
# Removing unwanted cols - reduce memory size
cols_to_remove = c('mid' , 'venue' , 'batsman', 'bowler', 'striker', 'non.striker', 'date')
df <- df %>% select(-cols_to_remove)

dim(df)
unique(df$bat_team)
# only keep current team which are present
consistent_team = c('Kolkata_Knight_Riders', 'Chennai_Super_Kings', 
'Rajasthan_Royals',
 'Mumbai_Indians','Kings_XI_Punjab',
 'Royal_Challengers_Bangalore', 'Delhi_Daredevils','Sunrisers_Hyderabad')
# filtering based on consistency
df <- df[ ( df$bat_team %in% consistent_team & df$bowl_team %in% consistent_team), 
]
# printing out unique team after filtering
unique(df$bat_team)
unique(df$bowl_team)
df <- df[ (df$overs >= 5.0), ]
head(df)
# install.packages("fastDummies")
# Load the library
library(fastDummies)
df <- dummy_cols(df, select_columns = c("bat_team", "bowl_team"))
cols_to_remove = c("bat_team", "bowl_team")
df <- df %>% select(-cols_to_remove)
head(df, 2)


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
#Linear Regression
independentvariables=colnames(df[,2:22])
independentvariables
Model=paste(independentvariables,collapse="+")
Model
Model=paste("total~",Model)
formula=as.formula(Model)
formula
slr <- lm(formula = formula, train)
summary(slr)
pred <- predict(slr, newdata = test)
mse_slr <- mean((test$total - pred)^2)


acc_slr <- 1 - mean( (abs(pred - test$total) / test$total) ) 
# Ridge
# install.packages("glmnet")
library(glmnet)
x.train <- train %>% select(-c("total"))
y.train <- train %>% select(c("total"))
x.test <- test %>% select(-c("total"))
y.test <- test %>% select(c("total"))
x.train <- as.matrix(x.train[,])
y.train <- as.matrix(y.train[,])
x.test <- as.matrix(x.test[,])
y.test <- as.matrix(y.test[,])
# Ridge 
alpha0.fit <- cv.glmnet(x.train, y.train, type.measure="mse", 
 alpha=0, family="gaussian")
alpha0.predicted <- predict(alpha0.fit, s=alpha0.fit$lambda.1se, newx=x.test)
mse_ridge <- mean((y.test - alpha0.predicted)^2)
acc_ridge <- 1 - mean( (abs(alpha0.predicted - y.test) / y.test) ) 
# Lasso
alpha1.fit <- cv.glmnet(x.train, y.train, type.measure="mse", 
 alpha=1, family="gaussian")
 alpha1.predicted <- predict(alpha1.fit, s=alpha1.fit$lambda.1se, newx=x.test)

mse_lasso <- mean((y.test - alpha1.predicted)^2)
acc_lasso <- 1 - mean( (abs(alpha1.predicted - y.test) / y.test) ) 
print(paste("Accuracy for SLR: ", acc_slr))
print(paste("Accuracy for Ridge: ", acc_ridge))
print(paste("Accuracy for Lasso: ", acc_lasso))


