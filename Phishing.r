library(ROSE)
library(ggplot2)
library(kableExtra)
library(lares)
library(plyr)
library(reshape2)
library(digest)
library(party)
library(rpart)
library(e1071)
library(caret)
library(rpart.plot)
library(tidyverse)
library(arules)
library(arulesViz)
library(rattle)
library(tree)
library(MASS)
library(caTools)
library(crosstable)
library(gmodels)
library(randomForest)
library(ROCR)
library(pROC)


Data_train = read.csv("Data.csv")

Data_test = read.csv("Data_old.csv")



Data_train = na.omit(Data_train)
Data_train <- subset (Data_train, select = -id)
names = c(1:31)

Data_train[,names] <- lapply(Data_train[,names] , factor)
str(Data_train)

Data_test = na.omit(Data_test)
Data_test <- subset (Data_test, select = -id)
names = c(1:31)

Data_test[,names] <- lapply(Data_test[,names] , factor)
str(Data_test)


Data_final= rbind(Data_train,Data_test)

Data_final


###############################33
table(Data_final$Result)

data.balanced.ou <- ovun.sample(Result~., data=Data_final,
                                N=nrow(Data_final), p=0.5, 
                                seed=1, method="both")$data



table(data.balanced.ou$Result)

# balanced data set with over-sampling
data.balanced.over <- ovun.sample(Result~., data=Data_final, 
                                  p=0.5, seed=1, 
                                  method="over")$data

data.balanced.under <- ovun.sample(Result~., data=Data_final, 
                                   p=0.5, seed=1, 
                                   method="under")$data
table(data.balanced.over$Result)

table(data.balanced.under$Result)

#Since over sampling returns better results we take the over sampling data

#The training accuracy of this data data set is high.

#removing observations may cause the training data to lose important information pertaining to majority class.



#Final Data Set to be used is "data.balanced.over" acheieved from over sampling

Final_Dataset = data.balanced.over


set.seed(1234)
Train_test_data <- sample.split(Y = Final_Dataset$Result, SplitRatio = 0.8)
train_dt <- Final_Dataset[Train_test_data,]
test_dt <- Final_Dataset[!Train_test_data,]

#########################################Building A Normal Decision Tree#######################################################
#Decision Tree
Decision_Tree = rpart(Result~., data = train_dt)
rpart.plot(Decision_Tree)

# Full Depth Tree
Full_Tree = rpart(Result~., data = train_dt, parms = list(split = "information"), control = rpart.control(minsplit = 0, minbucket = 0, cp = -1))
rpart.plot(Full_Tree)


#model building
#Gini Model
Gini_model <- rpart(Result~., data = train_dt, method = "class", parms = list(split="gini"))
plot(Gini_model$cptable)
print(Gini_model$cptable)
rpart.plot(Gini_model)

#Info Model
Info_model <- rpart(Result~., data = train_dt, method = "class", parms = list(split="information"))
plot(Info_model$cptable)
print(Info_model$cptable)
rpart.plot(Info_model)

#We get cp=0.01 from both Information Gain and Gini Of the model
#there is not much of a difference when we use Gini or information gain to construct the tree, hence we are taking gini 
#as our final variable.
#Gini_model$cptable expression list outs the cp values for the tree. 
#We select out the minimum CP value for which the xerror is minimum.


# for parameter tuning of decision tree,
# we are using two parameters to tune decision tree namely cp value and minsplit. we have fixed cp value to 0.01(obtained from model) 
#and we are running a for loop for multiple time to obtain optimum minsplit value. 
#function for accuracy prediction.............

##########
accuracy_tune <- function(var){
  predict_withdata <- predict(var, test_dt, type = "class")
  freq <- table(predict_withdata, test_dt$Result)
  accuracy <- sum(diag(freq))/sum(freq)
  accuracy
}


##############for loop for determine accuracy for large range min split for Decision Tree#############################

paratunerange <- c(0:50)
count<-1
accuracy.vector<-c()
for (i in paratunerange) {
  
  Gini_model <- rpart(Result~., data = train_dt , method = "class", control = rpart.control(minsplit = i,cp = 0.01))
  accuracy.vector[count] <- accuracy_tune(Gini_model)
  count<-count+1
}
accuracy.vector

plot(paratunerange,accuracy.vector)
lines(accuracy.vector)

max <-max(accuracy.vector)
index = which(accuracy.vector==max)+19
index
max

####### This suggest that this tree gives out maximum accuracy at minsplit values of 20,21,22,23,24,25,...,70.
### max accuracy is 89.89%

######################Using Cross Validation Technique to determine Accuracy of Decision Tree#############################################33

crossdata<-Data_final[sample(nrow(Final_Dataset)),]
k <- 10
nmethod <- 1
folds <- cut(seq(1,nrow(Final_Dataset)),breaks=k,labels=FALSE)
models.err <- matrix(-1,k,nmethod, dimnames=list(paste0("Fold", 1:k), c("rpart")))

for(i in 1:k)
{
  testIndexes <- which(folds==i, arr.ind=TRUE)
  test_DT <- Final_Dataset[testIndexes, ]
  train_DT <- Final_Dataset[-testIndexes, ]
  
  Valid_pointer <- sample(2, nrow(train_DT), replace = T, prob = c(0.8, 0.2))
  train_CV <- train_DT[Valid_pointer == 1, ]
  Validation_CV <- train_DT[Valid_pointer == 2, ]
  
  err <- c()
  Valid_rpart <- rpart(Result~., data = train_CV, method="class", control = rpart.control(minsplit = 10, cp = 0.01))
  predicted <- predict(Valid_rpart, newdata = Validation_CV, type = "class")
  err <- c(err,mean(Validation_CV$Result != predicted))
}
rpart.plot(Valid_rpart)
mean(err)

#Confusion Matrix for evaluating the model on training dataset
Predict_Train_tree <- predict(Valid_rpart,train_dt,type ="class")
#The accuracy of decision tree model model_tree is 90.23%.
confusionMatrix(Predict_Train_tree,train_dt$Result)


#Confusion Matrix for evaluating the model on test dataset
Predict_Test_tree <- predict(Valid_rpart,test_dt,type ="class")
#The accuracy of decision tree model model_tree is 90.17%.
confusionMatrix(Predict_Test_tree,test_dt$Result)

#This cross validation suggest that we have received 9% error from 5 fold cross validation which is in tune with previous accuracy obtained with decision tree.




#########################################ROC CURVE FOR DECISION TREE#################################################
pred_dt = predict(Valid_rpart, test_dt , type = "prob")
pred_tree = prediction(pred_dt[,2], test_dt$Result, label.ordering = c("1","-1"))
Roc_dt = performance(pred_tree, "tpr", "fpr")

plot(Roc_dt,colorize=T,
     main="ROC Curve for Decision Tree"
)

abline(a=0, b=1)

auc_dt <-performance(pred_tree,"auc")
auc_dt <-unlist(slot(auc_dt,"y.values"))
auc_dt

minauc_dt = min(round(auc_dt, digits = 2))
maxauc_dt = max(round(auc_dt, digits = 2))
minauct_dt = paste(c("min(AUC) = "), minauc_dt, sep = "")
maxauct_dt = paste(c("max(AUC) = "), maxauc_dt, sep = "")
legend(0.7, 0.5, c(minauct_dt, maxauct_dt, "\n"), border = "white", cex = 0.6, box.col = "blue")
abline(a= 0, b=1)


#################Gain Chart for DT##########################################################################


Gain_chart_DT <- performance(pred_tree,"rpp","tpr")
plot(Gain_chart_DT,
     colorize=T,
     main="Gain Chart for Decision Tree"
)

g_dt <-unlist(slot(Gain_chart_DT,"y.values"))
g_dt

minauc_g_dt = min(round(g_dt, digits = 2))
maxauc_g_dt = max(round(g_dt, digits = 2))
minauct_g_dt = paste(c("min(Gain) = "), minauc_g_dt, sep = "")
maxauct_g_dt = paste(c("max(Gain) = "), maxauc_g_dt, sep = "")
legend(0.8, 0.2, c(minauct_g_dt, maxauct_g_dt, "\n"), border = "white", cex = 0.6, box.col = "blue")

################Response Chart for DT #######################################################################
Response_Chart_DT <- performance(pred_tree,"rpp","ppv")
plot(Response_Chart_DT,
     colorize=T,
     main="Response Chart for Decision Tree"
)

r_dt <-unlist(slot(Response_Chart_DT,"y.values"))
r_dt

minauc = min(round(r_dt, digits = 2))
maxauc = max(round(r_dt, digits = 2))
minauct = paste(c("min(AUC) = "), minauc, sep = "")
maxauct = paste(c("max(AUC) = "), maxauc, sep = "")
legend(0.8, 0.2, c(minauct, maxauct, "\n"), border = "white", cex = 0.6, box.col = "blue")

######################PR Curve for DT#######################################################################
PR_DT= performance(pred_tree,"prec","rec")
plot(PR_DT,
     colorize=T,
     main="PR Chart for Decision Tree"
)

prr_dt <-unlist(slot(PR_DT,"x.values"))
prr_dt

minauc_prr_dt = min(round(prr_dt, digits = 2))
maxauc_prr_dt = max(round(prr_dt, digits = 2))
minauct_prr_dt = paste(c("min(PR) = "), minauc_prr_dt, sep = "")
maxauct_prr_dt = paste(c("max(PR) = "), maxauc_prr_dt, sep = "")
legend(0.8, 0.7, c(minauct_prr_dt, maxauct_prr_dt, "\n"), border = "white", cex = 0.6, box.col = "blue")
################################Naive Bayes_Code######################################################

naive_e1071 = naiveBayes(Result~., data = train_dt)
naive_e1071

#Prediction on the dataset
Predictions_naive=predict(naive_e1071,test_dt)

#Confusion matrix to check accuracy
table(Predictions_naive,test_dt$Result)
confusionMatrix(Predictions_naive,test_dt$Result)

################################Naive Bayes Cross Validation#################################################
library(klaR)
train_nb = Final_Dataset[,1:30]
test_nb = Final_Dataset$Result

# summarize results
model_cv = train(train_nb, test_nb, 'nb', trControl = trainControl(method='cv', number=10))
model_cv
pred_cv_nb=predict(model_cv,train_nb)
table(pred_cv_nb,test_nb)

confusionMatrix(pred_cv_nb, test_nb)


#########################################ROC CURVE FOR Naive Bayes#################################################

pred_roc_nb = predict(model_cv, test_dt, type = "prob")
pred_nb_pred = prediction(pred_roc_nb[,2], test_dt$Result, label.ordering = c("1","-1"))
Roc_nb = performance(pred_nb_pred, "tpr", "fpr")

plot(Roc_nb,colorize=T,
     main="ROC Curve for Naive Bayes"
)

abline(a=0, b=1)

auc_nb <-performance(pred_nb_pred,"auc")
auc_nb <-unlist(slot(auc_nb,"y.values"))
auc_nb

minauc_nb = min(round(auc_nb, digits = 2))
maxauc_nb = max(round(auc_nb, digits = 2))
minauct_nb = paste(c("min(AUC) = "), minauc_nb, sep = "")
maxauct_nb = paste(c("max(AUC) = "), maxauc_nb, sep = "")
legend(0.7, 0.5, c(minauct_nb, maxauct_nb, "\n"), border = "white", cex = 0.6, box.col = "blue")
abline(a= 0, b=1)

#################Gain Chart for Naive##########################################################################


Gain_chart_nb <- performance(pred_nb_pred,"rpp","tpr")
plot(Gain_chart_nb,
     colorize=T,
     main="Gain Chart for Naive Bayes"
)
g_nb <-unlist(slot(Gain_chart_nb,"y.values"))

minauc_g_nb = min(round(g_nb, digits = 2))
maxauc_g_nb = max(round(g_nb, digits = 2))
minauct_g_nb = paste(c("min(Gain) = "), minauc_g_nb, sep = "")
maxauct_g_nb = paste(c("max(Gain) = "), maxauc_g_nb, sep = "")
legend(0.8, 0.2, c(minauct_g_nb, maxauct_g_nb, "\n"), border = "white", cex = 0.6, box.col = "blue")

################Response Chart for Naive #######################################################################
Response_Chart_nb <- performance(pred_nb_pred,"rpp","ppv")
plot(Response_Chart_nb,
     colorize=T,
     main="Response Chart for Naive Bayes"
)

######################PR Curve for Naive##########################################################################
PR_nb= performance(pred_nb_pred,"prec","rec")
plot(PR_nb,
     colorize=T,
     main="PR Chart for Naive Bayes"
)

prr_nb <-unlist(slot(PR_nb,"x.values"))

minauc_prr_nb = min(round(prr_nb, digits = 2))
maxauc_prr_nb = max(round(prr_nb, digits = 2))
minauct_prr_nb = paste(c("min(PR) = "), minauc_prr_nb, sep = "")
maxauct_prr_nb = paste(c("max(PR) = "), maxauc_prr_nb, sep = "")
legend(0.8, 0.7, c(minauct_prr_nb, maxauct_prr_nb, "\n"), border = "white", cex = 0.6, box.col = "blue")

#############################################################################################################################
#################Random Forest##################################################################################
library(randomForest)
model_rf <- randomForest(Result~.,data=train_dt,ntree=50,mtry=2,importance=TRUE,proximity=TRUE)

#Testing on Training dataset
#Predictions on training dataset

Predict_Train_rf<-predict(model_rf,train_dt,type = "class")

#Confusion matrix for evaluating the model on training dataset
confusionMatrix(Predict_Train_rf,train_dt$Result)
#The Accuracy of Random Forest on Training Data is 95.92%.



#Predicting for Test Data

Predict_Test_rf = predict(model_rf, test_dt, type = "class")

confusionMatrix(Predict_Test_rf, test_dt$Result)
#The Accuracy for Random Forest on Test Data is 95.71%.

#Extracting single tree in Random Forest.
getTree(model_rf, 1, labelVar = TRUE)

#################################Random Forest Using Cross Validation################################################################################

k =10
nmethod = 1
folds = cut(seq(1,nrow(Final_Dataset)),breaks=k,labels=FALSE)
models.err = matrix(-1,k,nmethod,dimnames=list(paste0('Fold', 1:k), c('rf')))
for(i in 1:k)
{
  testIndexes = which(folds==i, arr.ind=TRUE)
  testData = Final_Dataset[testIndexes, ]
  trainData = Final_Dataset[-testIndexes,]
  
  rf = randomForest(Result ~., data = train_dt, ntree = 100, mtry = sqrt(ncol(train_dt) - 1))
  predicted = predict(rf, newdata = test_dt, type="class")
  e <- ifelse(predicted == test_dt$Result, 1, -1)
  mean_rf = sum(e) / nrow(test_dt)
  models.err[i] <- mean_rf

}


print(models.err)

predicted = predict(rf, newdata = test_dt, type="class")

confusionMatrix(predicted, test_dt$Result)


################################################################################################################################
#########################################ROC CURVE FOR RF############################################################

pred_rf = predict(rf, test_dt , type = "prob")
pred_rf_pred = prediction(pred_rf[,2], test_dt$Result, label.ordering = c("1","-1"))
Roc_rf = performance(pred_rf_pred, "tpr", "fpr")

plot(Roc_rf,colorize=T,
     main="ROC Curve for Random Forest"
)

abline(a=0, b=1)

auc <-performance(pred_rf_pred,"auc")
auc <-unlist(slot(auc,"y.values"))
auc

minauc = min(round(auc, digits = 2))
maxauc = max(round(auc, digits = 2))
minauct = paste(c("min(AUC) = "), minauc, sep = "")
maxauct = paste(c("max(AUC) = "), maxauc, sep = "")
legend(0.7, 0.5, c(minauct, maxauct, "\n"), border = "white", cex = 0.6, box.col = "blue")
abline(a= 0, b=1)


################################################Gain Chart for RF##################################################################

Gain_chart_rf <- performance(pred_rf_pred,"rpp","tpr")
plot(Gain_chart_rf,
     colorize=T,
     main="Gain Chart for Random Forest"
)

g_rf <-unlist(slot(Gain_chart_rf,"y.values"))
g_rf

minauc_g_rf = min(round(g_rf, digits = 2))
maxauc_g_rf = max(round(g_rf, digits = 2))
minauct_g_rf = paste(c("min(Gain) = "), minauc_g_rf, sep = "")
maxauct_g_rf = paste(c("max(Gain) = "), maxauc_g_rf, sep = "")
legend(0.8, 0.2, c(minauct_g_rf, maxauct_g_rf, "\n"), border = "white", cex = 0.6, box.col = "blue")

################Response Chart for RF #######################################################################
Response_Chart_rf <- performance(pred_rf_pred,"rpp","ppv")
plot(Response_Chart_rf,
     colorize=T,
     main="Response Chart for Random Forest"
)

######################PR Curve for RF#######################################################################
PR_rf= performance(pred_rf_pred,"prec","rec")
plot(PR_rf,
     colorize=T,
     main="PR Chart for Random Forest"
)

prr_rf <-unlist(slot(PR_rf,"x.values"))
prr_rf

minauc_prr_rf = min(round(prr_rf, digits = 2))
maxauc_prr_rf = max(round(prr_rf, digits = 2))
minauct_prr_rf = paste(c("min(PR) = "), minauc_prr_rf, sep = "")
maxauct_prr_rf = paste(c("max(PR) = "), maxauc_prr_rf, sep = "")
legend(0.8, 0.7, c(minauct_prr_rf, maxauct_prr_rf, "\n"), border = "white", cex = 0.6, box.col = "blue")

##############################Comparing ROC CURVES################################################################
#Comparing the ROC curves for Random Forest, Naive Bayes and Decision tree

par(mfrow=c(1,3))
plot(Roc_rf,
     colorize=T,
     main="ROC Curve for Random Forest"
)
abline(a=0, b=1)
legend(0.7, 0.5, c(minauct, maxauct, "\n"), border = "white", cex = 0.5, box.col = "blue")


plot(Roc_dt,
     colorize=T,
     main="ROC Curve for Decision Tree"
)
abline(a=0, b=1)
legend(0.7, 0.5, c(minauct_dt, maxauct_dt, "\n"), border = "white", cex = 0.5, box.col = "blue")

plot(Roc_nb,
     colorize=T,
     main="ROC Curve for Naive Bayes"
)
abline(a=0, b=1)
legend(0.7, 0.5, c(minauct_nb, maxauct_nb, "\n"), border = "white", cex = 0.5, box.col = "blue")

##############################Comparing Gain Charts################################################################
#Comparing the Gain Charts for Random Forest, Naive Bayes and Decision tree

par(mfrow=c(1,3))
plot(Gain_chart_rf,
     colorize=T,
     main="Gain Chart for Random Forest"
)
abline(a=0, b=1)
legend(0.7, 0.5, c(minauct_g_rf, maxauct_g_rf, "\n"), border = "white", cex = 0.5, box.col = "blue")


plot(Gain_chart_DT,
     colorize=T,
     main="Gain Chart for Decision Tree"
)
abline(a=0, b=1)
legend(0.7, 0.5, c(minauct_g_dt, maxauct_g_dt, "\n"), border = "white", cex = 0.5, box.col = "blue")

plot(Gain_chart_nb,
     colorize=T,
     main="Gain Chart for Naive Bayes"
)
abline(a=0, b=1)
legend(0.7, 0.5, c(minauct_g_nb, maxauct_g_nb, "\n"), border = "white", cex = 0.5, box.col = "blue")

##############################Comparing PR Charts################################################################
#Comparing the PR Charts for Random Forest, Naive Bayes and Decision tree

par(mfrow=c(1,3))
plot(PR_rf,
     colorize=T,
     main="PR Chart for Random Forest"
)
abline(a=0, b=1)
legend(0.7, 0.6, c(minauct_prr_rf, maxauct_prr_rf, "\n"), border = "white", cex = 0.5, box.col = "blue")


plot(PR_DT,
     colorize=T,
     main="PR Chart for Decision Tree"
)
abline(a=0, b=1)
legend(0.7, 0.6, c(minauct_prr_dt, maxauct_prr_dt, "\n"), border = "white", cex = 0.5, box.col = "blue")

plot(PR_nb,
     colorize=T,
     main="PR Chart for Naive Bayes"
)
abline(a=0, b=1)
legend(0.7, 0.6, c(minauct_prr_nb, maxauct_prr_nb, "\n"), border = "white", cex = 0.5, box.col = "blue")
#########################################################################################################################################
# For testing the results obtained, we used 3 parameters: Accuracy, Recall and False Positive Rate (FPR).
# Accuracy: It is the ratio of number of correct predictions to the total number of input samples.
# Since we want most of the URLs to be classified correctly, hence we require high accuracy.
# 
# Recall: It is the ratio of number of true positives to the total number of predicted positives. 
# As we want most of our websites to be predicted as positive, to be legitimate, high recall is desired.
# For showing this comparison, we are using PR (Precision-Recall) Chart for all the three models.
# 
# False Positive Rate (FPR): It is the ratio of number of samples incorrectly identified as positive to to total number of actually negative samples. 
# We need to minimize the number of phishing websites identified as legitimate as it can lead to unwanted problems/losses for the person visiting the website. 
# Thus, low FPR is one of the measure that's required here. For showing this comparison, we have used ROC Chart for all the three models.

########################################################################################################################################
#############################################Final Analysis#############################################################################
# We have created models for Decision Tree, Naive Bayes and Random Forest, checked its performance using  10 fold Cross Validation measure.

#After the conducting the 10 fold Cross Validation in each of the models(Decision Tree, Naive Bayes and Random Forest), we found that 
#Random Forest has the highest accuracy.To tackle Phishing websites, we require a measure of both True Positive Rate and False Positive 
#Rate. We have shown accuracy comparison between our models through the ROC curve, the Gain Chart, the Response Chart and the Precision Recall Curve. 
#But we choose the ROC curve as our final evaluation measure as it has both True Positive Rate and False Positive Rate which is utmost important for 
#identifying the phishing and the legitimate websites. We plotted the ROC curve for all the three models(Decision Tree, Naive Bayes and Random Forest) and 
#we got the below AUC accuracy measure:

#Decision Tree: 93.5%
#Naive Bayes:   84.41%
#Random Forest: 99.7%

#Thus we choose the model of Random Forest with the evaluation metric as ROC curve.







