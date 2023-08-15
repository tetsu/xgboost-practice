#loading data
data <- read.csv("bank-full.csv", sep = ";")

#look at the data structure
str(data)

#creating dataset with numerical variables only
#install.packages("dplyr")
library(dplyr)
dataset_numerical <- data %>% select_if(is.numeric)

#summary statistics and correlation matrix
summary(dataset_numerical)
cor(dataset_numerical)

#adding dependent variable to dataset
dataset_numerical <- cbind(dataset_numerical, data$y)
colnames(dataset_numerical)[8] <- "yes"

#splitting dataset into training and test set
#install.packages("caTools")
library(caTools)
set.seed(1502)
split <- sample.split(dataset_numerical$yes, SplitRatio = 0.8)
training_set <- subset(dataset_numerical, split == TRUE)
test_set <- subset(dataset_numerical, split == FALSE)

#isolating y variable
train.y <- as.numeric(as.factor(training_set$yes)) - 1
test.y <- as.numeric(as.factor(test_set$yes)) - 1

#isolate the x variable
train.X <- as.matrix(training_set[, 2:ncol(training_set)-1])
test.X <- as.matrix(test_set[, 2:ncol(test_set)-1])

#state the parameters
#install.packages("xgboost")
library(xgboost)
parameters <- list(eta = 0.3,
                   max_depth = 2,
                   subsample = 1,
                   colsample_bytree = 1,
                   min_child_weight = 1,
                   gamma = 0,
                   set.seed = 1502, 
                   eval_metric = "auc",
                   objective = "binary:logistic",
                   booster = "gbtree")

#detecting how many cores we haves
#install.packages("doParallel")
library(doParallel)
detectCores()

#Running XGBoost
model1 <- xgboost(data = train.X,
                  label = train.y,
                  set.seed(1502),
                  nthread = 6,
                  nround = 200,
                  params = parameters,
                  print_every_n = 50,
                  early_stopping_rounds = 20,
                  verbose = 1)

#Predictions
predictions1 <- predict(model1, newdata = test.X)
predictions1 <- ifelse(predictions1 > 0.5, 1, 0)

#Checking accuracy
#install.packages("caret")
library(caret)
confusionMatrix(table(predictions1, test.y))
mean(test.y)

############################################################

#tranform the factors into dummy variables
#install.packages("fastDummies")
library(fastDummies)
dataset_dummy <- dummy_cols(data, remove_first_dummy = TRUE)

#get only the dummy variables
dataset_dummy <- dataset_dummy[, 18:ncol(dataset_dummy)]

#creating final dataset
dataset <- cbind(dataset_dummy, dataset_numerical)
dataset <- dataset %>% select(-y_yes)

#############################################################

#splitting dataset into training and test set part 2
library(caTools)
set.seed(1502)
split <- sample.split(dataset$yes, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

#isolating y variable part 2
train.y <- as.numeric(as.factor(training_set$yes)) - 1
test.y <- as.numeric(as.factor(test_set$yes)) - 1

#isolate the x variable par 2
train.X <- as.matrix(training_set[, 2:ncol(training_set)-1])
test.X <- as.matrix(test_set[, 2:ncol(test_set)-1])

#Running XGBoost part 2
library(xgboost)
model2 <- xgboost(data = train.X,
                  label = train.y,
                  set.seed(1502),
                  nthread = 6,
                  nround = 200,
                  params = parameters,
                  print_every_n = 50,
                  early_stopping_rounds = 20,
                  verbose = 1)

#Predictions part 2
predictions2 <- predict(model2, newdata = test.X)
predictions2 <- ifelse(predictions2 > 0.5, 1, 0)

#Checking accuracy
#install.packages("caret")
library(caret)
confusionMatrix(table(predictions2, test.y))

##################################################################
#start parallel processing
library(doParallel)
cpu <- makeCluster(6)
registerDoParallel(cpu)

#state inputs
y <- as.factor(as.numeric(as.factor(dataset$yes)) - 1)
X <- as.matrix(dataset[, 2:ncol(dataset)-1])

#cross validation parameters
library(caret)
tune_control <- trainControl(method = "cv",
                             number = 5,
                             allowParallel = TRUE)

#setting the parameter grid
tune_grid <- expand.grid(nrounds = seq(200, to = 1800, by = 200),
                         eta = c(0.05, 0.3),
                         max_depth = c(2,4,6,8),
                         subsample = c(0.9, 1),
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         gamma = 0)

#cross validation and parameter tuning start
start <- Sys.time()
xgb_tune <- train(x = X,
                  y = y,
                  method = "xgbTree",
                  trControl = tune_control,
                  tuneGrid = tune_grid)
end <- Sys.time()

#retrieve the results
xgb_tune$bestTune


#######################################################################
#start parallel processing
library(doParallel)
cpu <- makeCluster(6)
registerDoParallel(cpu)

#setting the parameter grid part 2
tune_grid2 <- expand.grid(nrounds = seq(200, to = 1800, by = 200),
                         eta = xgb_tune$bestTune$eta,
                         max_depth = xgb_tune$bestTune$max_depth,
                         subsample = xgb_tune$bestTune$subsample,
                         colsample_bytree = c(0.5, 1),
                         min_child_weight = seq(1, to = 4, by = 1),
                         gamma = c(0, 0.05))

#cross validation and parameter tuning start part 2
start <- Sys.time()
xgb_tune2 <- train(x = X,
                  y = y,
                  method = "xgbTree",
                  trControl = tune_control,
                  tuneGrid = tune_grid2)
end <- Sys.time()

#retrieve the results part 2
xgb_tune2$bestTune

#######################################################################

#set the tuned parameters
parameters_tune <- list(eta = xgb_tune2$bestTune$eta,
                   max_depth = xgb_tune2$bestTune$max_depth,
                   subsample = xgb_tune2$bestTune$subsample,
                   colsample_bytree = xgb_tune2$bestTune$colsample_bytree,
                   min_child_weight = xgb_tune2$bestTune$min_child_weight,
                   gamma = xgb_tune2$bestTune$gamma,
                   set.seed = 1502, 
                   eval_metric = "auc",
                   objective = "binary:logistic",
                   booster = "gbtree")

#running xgboost for the final time
model3 <- xgboost(data = train.X,
                  label = train.y,
                  set.seed(1502),
                  nthread = 6,
                  nround = xgb_tune2$bestTune$nrounds,
                  params = parameters_tune,
                  print_every_n = 50,
                  early_stopping_rounds = 20,
                  verbose = 1)

#Predictions part 3
predictions3 <- predict(model3, newdata = test.X)
predictions3 <- ifelse(predictions3 > 0.05, 1, 0)

#Checking accuracy
confusionMatrix(table(predictions3, test.y))

#importance drivers
importance <- xgb.importance(feature_names = colnames(test.X),
                             model = model3)
xgb.plot.importance(importance_matrix = importance)

#shap values
xgb.plot.shap(data = test.X,
              model = model3,
              top_n = 3)






