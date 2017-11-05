#Tutorial
#https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/
#https://www.analyticsvidhya.com/blog/2016/09/this-machine-learning-project-on-imbalanced-data-can-add-value-to-your-resume/

#imbalanced data is the data with one class (dependent variable) outperforms others
#as a result, ML algorithms will fail to catch the pattern for the minority class due to not-enough data
#generally, minority would be wrongly classified as majority
#Solution:
#Undersampling
#Oversampling
#Synthetic Data Generation
#Cost Sensitive Learning

#1.undersampling
#reduce (undersample) majority data to make it balance. Need huge data set.
#Random undersampling:randomly choose from majority data
#Informative undersampling1: EasyEnsemble, samples n subsets from majortity, build n classifiers for each (majority n + minority) set 
#Informative undersampling2: Ensemble classifiers and systematically selects what majority n to ensemble based on criteria(supervised)
#Cons: may lead to imformation loss for small data set

#2. Oversampling
#oversample minority
#random sampling:random oversample
#informative:use criteria to conduct sampling
#PROS: no information loss
#CONS: simple repitation of same pattern will lead to overfitting (high variance). Bad for unseen data.

#3. Both sampling
#conbine over/under sampling to overcome overfitting and information loss

#4. Synthetic data generation:synthetic minority oversampling technique (SMOTE)
#each of N new samples: i-th feature in one minority sample + diff(feature_i in this sample,feature_i in one KNN sample)*random(0,1)_i 

#5. COst Sensitive Learning (CSL)
#no sampling. It up weights FN/FP for learning (like boosted trees)

#Metric selection
#For imbalanced data set we cannot use simply accuracy/PPV/NPV (would be biased by majority)
#sensitivity/specificity are recommended

library(ROSE)
library(ggplot2)
library(magrittr)
library(caret)
data(hacide)
str(hacide.train)
head(hacide.train)
#easy to see it is unbalanced
table(hacide.train$cls)
prop.table(table(hacide.train$cls))
as.data.frame(table(hacide.train$cls)) %>% ggplot(aes(x=Var1,y=Freq,fill=Var1)) + geom_bar(stat="identity")

#In this case, 0:P (majority), 1:N (minority)
#decision tree only reports specificity of 0.2 almost unable to dected TN
#Any metric that contain TP, FN would be biased since majority is unlikely to be wrong
model <- hacide.train %>% train(cls~.,data=.,method="rpart")
pred <- hacide.test %>% predict(model,newdata = .)
confusionMatrix(pred,hacide.test$cls)
#also, AUC of ROC is very low
roc.curve(hacide.test$cls, pred, plotit = T)

#Oversampling so P:N~=1:1
data_balanced_over <-  ovun.sample(cls~.,data=hacide.train,method="over")
table(data_balanced_over$data$cls)
#Undersampling so P:N~=1:1
data_balanced_under <-  ovun.sample(cls~.,data=hacide.train,method="under")
table(data_balanced_under$data$cls)
#Both oversampling and undersampling
data_balanced_both <-  ovun.sample(cls~.,data=hacide.train,method="both")
table(data_balanced_both$data$cls)
#Synthetic data generation
data.rose <- ROSE(cls~.,data=hacide.train,seed=1)
table(data.rose$data$cls)

#Test
model_over <- data_balanced_over$data %>% train(cls~.,data=.,method="rpart")
pred_over <- hacide.test %>% predict(model_over,newdata = .)

model_under <- data_balanced_under$data %>% train(cls~.,data=.,method="rpart")
pred_under <- hacide.test %>% predict(model_under,newdata = .)

model_both <- data_balanced_both$data %>% train(cls~.,data=.,method="rpart")
pred_both <- hacide.test %>% predict(model_both,newdata = .)

model_rose <- data.rose$data %>% train(cls~.,data=.,method="rpart")
pred_rose <- hacide.test %>% predict(model_rose,newdata = .)

confusionMatrix(pred_over,hacide.test$cls)#we can see overfitting for "1" (Negative)
confusionMatrix(pred_under,hacide.test$cls)
confusionMatrix(pred_both,hacide.test$cls)
confusionMatrix(pred_rose,hacide.test$cls)#SMOTE is the best

roc.curve(hacide.test$cls, pred_over, plotit = T,col="red")
roc.curve(hacide.test$cls, pred_under, plotit = T,add=TRUE,col="purple")
roc.curve(hacide.test$cls, pred_both, plotit = T,add=TRUE,col="orange")
roc.curve(hacide.test$cls, pred_rose, plotit = T,add=TRUE,col="blue")#SMOTE is the best
