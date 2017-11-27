#a very new ML package
library(mlr)
library(magrittr)
library(ggplot2)
library(dplyr)
library(reshape2)
#Hypothesis generation
#Ho: There exits No significant impact of indipendent variables on the dependent variable (income -50k ? +50k?)
#Ha: There exits significant impact of indipendent variables on the dependent variable
library(data.table)
#"not in universe" != "NA" they both appear in train dataset
train <- read.csv("train.csv",header = TRUE,na.strings = c(" ?","NA"))
test <- read.csv("test.csv",header = TRUE,na.strings = c(" ?","NA"))

#################################################EDA########################################
dim(train)#41 features, this is a big number
dim(test)#accounts for almost 70%

View(train)#For editing use edit(train)
View(test)
#income_level show have different levels for "+50000" in train and test
#need to unify
train <- train %>% mutate(income_level = ifelse(income_level==-50000,"low","high"))
test <- test %>% mutate(income_level = ifelse(income_level==-50000,"low","high"))

str(train)

summary(train)

head(train,5);head(test,5)

#check target feature
train$income_level %>% table() %>% prop.table() #very imbalanced:majority (93.8%): low

#clean feature type (factor/numeric) according to data codebook
factorCol <- c(2:5,7,8:16,20:29,31:38,40,41)
numericCol <- seq(1,41,1)[!seq(1,41,1) %in% factorCol]

train[,factorCol] <- train[,factorCol] %>% apply(2,factor)
train[,numericCol] <- train[,numericCol] %>% apply(2,as.numeric)
test[,factorCol] <- test[,factorCol] %>% apply(2,factor)
test[,numericCol] <- test[,numericCol] %>% apply(2,as.numeric)

#split numerical and categorical data for analysis
#why?because sapply eventually returns a matrix with same data type, which alters original types
train_num <- train[,numericCol]
train_factor <- train[,factorCol]
test_num <- test[,numericCol]
test_factor <- test[,factorCol]

#discover numeric variable distribution via shapiro test (H0:normal)
shapiro_p <- function(x){
  if(length(x)<=5000){
    return(ifelse(shapiro.test(x)$p.value <= 0.05,"Nonnormal","Normal"))
  }else{
    return(ifelse(shapiro.test(x[sample(seq(1,length(x),1),5000,replace = FALSE)])$p.value <= 0.05,"Nonnormal","Normal"))
  }
}

#No numerical feature is normally distributed!!!
train_num_dis <- data.frame(Var=colnames(train_num),Normality=as.vector(sapply(train_num,shapiro_p)))

#visually check distribution using histgram
library(gridExtra)
library(plotly)
g1 <- train_num %>% ggplot(aes(age,y=..density..))+geom_histogram(bins = 100)+geom_density()
ggplotly(g1)
g2 <- train_num %>% ggplot(aes(wage_per_hour,y=..density..))+geom_histogram(bins = 100)+geom_density()
g3 <- train_num %>% ggplot(aes(capital_gains,y=..density..))+geom_histogram(bins = 100)+geom_density()
g4 <- train_num %>% ggplot(aes(capital_losses,y=..density..))+geom_histogram(bins = 100)+geom_density()
g5 <- train_num %>% ggplot(aes(dividend_from_Stocks,y=..density..))+geom_histogram(bins = 100)+geom_density()
g6 <- train_num %>% ggplot(aes(num_person_Worked_employer,y=..density..))+geom_histogram(bins = 100)+geom_density()
g7 <- train_num %>% ggplot(aes(weeks_worked_in_year,y=..density..))+geom_histogram(bins = 100)+geom_density()
grid.arrange(g1, g2, g3, g4, g5, g6, g7,ncol=3)
#Theoretically we should take log for normality


#also, study dependent variable vs independent variable (presented by color)
#First, numerical variables
train_num <- train_num %>% cbind((train_factor %>% select(income_level)))
#Most high-income people are roughly between 25 and 70
#High wage/hour (contracting?) doesn't mean high total income
train_num %>% ggplot(aes(x=age,y=wage_per_hour,color=income_level))+geom_point()
#the more people worked for, the higher prop to have income
train_num %>% ggplot(aes(x=num_person_Worked_employer,y=wage_per_hour,color=income_level))+geom_point()
#Then, categorical variables
#barplot for x is count(like hist) plot split by fill
#dominated by "not in universe" and "private"
#best practive:combine <5% variables
train_factor %>% ggplot(aes(x=class_of_worker,fill=income_level))+
  geom_bar(position = "dodge")+theme(axis.text.x = element_text(angle=60,hjust=1))
#education shows reasonable trend
train_factor %>% ggplot(aes(x=education,fill=income_level))+
  geom_bar(position = "dodge")+theme(axis.text.x = element_text(angle=60,hjust=1))
#use prop.table to show percentage
table(train_factor$education,train_factor$income_level) %>% apply(1,prop.table)
#to plot prop bar we need to manipulate the data frame...
#melt in reshape2 is super powerful here (melt high and low prop to (key,value))
#prop.table(1):1 calculates prop for each row
prop_table <- table(train_factor$education,train_factor$income_level) %>% prop.table(1) %>% as.data.frame()
names(prop_table) <- c("education","income_level","prop")
prop_table %>% ggplot(aes(x=education,y=prop,fill=income_level))+geom_bar(position = "dodge",stat = "identity")+theme(axis.text.x = element_text(angle=60,hjust=1))

#############################################Data Cleaning########################################
###for numerical features
#check NA
#luckily no NA
table(is.na(train_num))
table(is.na(test_num))
#check correlation for numerical features E[(X-mu_x)(Y-mu_y)]/(sigma_xsigma_y)
library(caret)
#check which correlated feature shall be removed based on cor >= 0.7
#in ML, correlated features theoretically have no negative impacts, however
#in practice, we have limited sample size, given this, high dimention of features would lead to curse of dimensionality(low accuracy)
#e.g. high dimention demands exponentially high sample size to obtain stat significance at each point (low representation level at each data point)
#besides, high dimention makes harder to find patterns
#so we need to reduce dimention (PCA, AutoEncoder, rfe, remove cor features...) given limited data points
train_cor <- findCorrelation(cor(train_num %>% select(-income_level)),cutoff = 0.7,names = FALSE)#return index
#why? see cor() we can see cor(weeks_worked_in_year,num_person_worked_employer) is 0.74
cor(train_num %>% select(-income_level))
#remove this high cor variable from train and test
train_num <- train_num %>% select(-train_cor)
test_num <- test_num %>% select(-train_cor)

###for categorical features(only handle NA since can't do cor)
#check NA percentile and remove features with high percentage (>5%) of NA values
#see here how to use match for extracting col index since select/subset(select) don't take character vector
NA_train_factor <- train_factor %>% apply(2,function(x){mean(is.na(x))*100}) %>% subset(.,.<5) %>% names() %>% match(names(train_factor))
NA_test_factor <- test_factor %>% apply(2,function(x){mean(is.na(x))*100}) %>% subset(.,.<5) %>% names() %>% match(names(test_factor))
train_factor <- train_factor %>% select(NA_train_factor)
test_factor <- test_factor %>% select(NA_test_factor)
#set NA to normal character
train_factor[is.na(train_factor)] <- "Unavailable"
test_factor[is.na(test_factor)] <- "Unavailable"
train_factor <- train_factor %>% apply(2,factor) %>% as.data.frame()
test_factor <- test_factor %>% apply(2,factor) %>% as.data.frame()

#############################################Data Manipulation/preprocessing######################################
#this is only for categorical data
#combine low-frequency factor levels (< 5%) into "others" since they might not show up in test sets
#this will increase the certainty and reduce noise of this feature
#e.g. in Decision tree, max gini index = 1-1/k, k is levels, k increases, max gini also increases so this feature is too noisy
#small Gini (1-certainty or purity) and high info gain (Entropy-Entropy_feature) both require high certainty or purity of the feature
#rememebr to unlevel and relevel
combine_5 <- function(x){
  x <- x %>% as.character()
  level_name <- names(table(x))[as.vector(prop.table(table(x))*100<5)]#5% threshold
  x <- ifelse(x %in% level_name,"others",x)
  x <- x %>% factor()#if not data.frame, final would be matrix
  return(x)
}
train_factor <- train_factor %>% apply(2,combine_5) %>% as.data.frame()
test_factor <- test_factor %>% apply(2,combine_5) %>% as.data.frame()

#check if test and train have same levels for features
#this time all good
library(mlr)
summarizeColumns(train_factor)$nlevs == summarizeColumns(test_factor)$nlevs

#next we do feature binning for numerics
#many ML algorithms work better with binned features that come with less noise but this leads to information loss
#preprocessing:centering, scaling(norm[0,1],stand[(x-mu)/sigma],log),binning, combine minority levels
#feature engineering:add feature(synthetic), remove feature(pca,cor,rfe,AutoEncoder)
#binning could be based on quantile
train_num %>% select(-income_level) %>% apply(2,quantile)

#age:c(young,adult,old);
#wage_per_hour:c(low,high);capital_gains/losses:c(low,high);dividend_from_stocks:c(low,high)
#the above features show strong binary pattern in histgram
#num_person_employer:no binning;weeks_worked_in_year:no binning
#use cut function
train_num$age <- train_num$age %>% cut(breaks=c(0,30,60,90),labels=c("young","adult","old"))
train_num$capital_gains <- ifelse(train_num$capital_gains==0,"zero","more") %>% factor()
train_num$capital_losses <- ifelse(train_num$capital_losses==0,"zero","more") %>% factor()
train_num$dividend_from_Stocks <- ifelse(train_num$dividend_from_Stocks==0,"zero","more") %>% factor()

test_num$age <- test_num$age %>% cut(breaks=c(0,30,60,90),labels=c("young","adult","old"))
test_num$capital_gains <- ifelse(test_num$capital_gains==0,"zero","more") %>% factor()
test_num$capital_losses <- ifelse(test_num$capital_losses==0,"zero","more") %>% factor()
test_num$dividend_from_Stocks <- ifelse(test_num$dividend_from_Stocks==0,"zero","more") %>% factor()
#remove income_level for next step
train_num <- train_num %>% select(-income_level)

#############################################ML modelling######################################
#try mlr here and compare with caret. Seems to have more functions even smote(). Supports some interfaces to caret.
library(mlr)
library(caret)

#accuracy is not a good metric for imbalanced data
#it is distorted by the majority data and loses the evaluation on minority
#first, combine categorical data and numerical data
train_DT <- cbind(train_num,train_factor)
test_DT <- cbind(test_num,test_factor)
#???test_DT$country mother has 3 levels but train_DT has two
summarizeColumns(train_DT)$nlevs == summarizeColumns(test_DT)$nlevs
#need to clean  the leading whitespace in factors test set (why in NB this is not required)
#this step shall be done in the very begining
#use sapply for dataframe, list. 
#The reason why num -> factor is become the retuned matrix can only contains one type of data.
test_DT <- test_DT %>% sapply(trimws) %>% as.data.frame()
test_DT$wage_per_hour <- as.numeric(test_DT$wage_per_hour)
test_DT$num_person_Worked_employer <- as.numeric(test_DT$wage_per_hour)
test_DT$industry_code <- as.character(test_DT$industry_code) %>% ifelse(.=="0"," 0",.) %>% as.factor()
test_DT$occupation_code <- as.character(test_DT$occupation_code) %>% ifelse(.=="0"," 0",.) %>% as.factor()
#"Mexico" is the extra level but it only accounts for 5.1%
prop.table(table(test_DT$country_mother))
#Decide to treat it as "others"
test_DT$country_mother <- test_DT$country_mother %>% as.character() %>% 
  ifelse(.==" United-States","United-States",.) %>% ifelse(.==" Mexico","others",.) %>% factor()

#create task
#caret:NA
#mlr:task encapsulates data and target variable and can access elements using getTaskXXX()
train.task.mlr <-makeClassifTask(data=train_DT,target = "income_level") 
test.task.mlr <- makeClassifTask(data=test_DT,target = "income_level") 

#remove variable with near zero variance
#caret
train.task.caret <- train_DT[,-nzv(train_DT)]
#mlr
train.task.mlr <- removeConstantFeatures(train.task.mlr,perc = 0.06)#at least 6% values differ from the mode to keep this feature
test.task.mlr <- removeConstantFeatures(test.task.mlr,perc = 0.06)#at least 6% values differ from the mode to keep this feature

#variance importance 
#doubt:shall we check it now? var_imp is based on trees using gini index/entropy gain. This won't reflect the truth for imbalanced data set.
#caret: NA (can do only after modelling varImp(model))
#mlr: supports many methods
var_imp.mlr <- generateFilterValuesData(train.task.mlr,method = c("information.gain","chi.squared"))
plotFilterValues(var_imp.mlr,feat.type.cols = TRUE)#color numerical feature

#SMOTE:already know that SMOTE >> over/under sampling
#caret: need SMOTE/ROSE package/or the built-in upSample(train_DT,train_DT$income_level)/ or sampling in trainControl
#ROSE:1.based prop to decide sampled class y_i 2.randomly find a record x_i 3. new x generated from pdf of x_i 
library(ROSE)
library(DMwR)
train_DT.under <- ovun.sample(income_level~.,data=train.task.caret,method="under",seed=1)$data
train_DT.over <- ovun.sample(income_level~.,data=train.task.caret,method="over",seed=1)$data
train_DT.rose <- ROSE(income_level~.,data=train.task.caret,seed=1)$data
prop.table(table(train.task.caret$income_level));prop.table(table(train_DT.over$income_level))
#perc.over in SMOTE:new majority = diff(new minority,old minority) if perc.under=100
#so perc.over=100*old majority/old moniroty -100; perc.under = (old majority/old moniroty)/(old majority/old moniroty-100) 
train_DT.smote <- SMOTE(income_level~.,data=train.task.caret,perc.over = 1411,perc.under = 107)
#mlr:need to set rate and nn
#rate: size of after sample / before sample, here we make two factors equal
rate_under <- table(train_DT$income_level)[1]/table(train_DT$income_level)[2]
train.under <- undersample(train.task.mlr,rate=rate_under)
prop.table(table(getTaskTargets(train.under)))#use getTaskData to view task data 

rate_over <- table(train_DT$income_level)[2]/table(train_DT$income_level)[1]
train.over <- oversample(train.task.mlr,rate=rate_over)
prop.table(table(getTaskTargets(train.over)))
#the smote here is kind of unreliable since education "children" is sampled to have "high" income
train.smote <- smote(train.task.mlr,rate_over)#longer time due to complexity
prop.table(table(getTaskTargets(train.smote)))


#Algorithm selection for binary classification
#caret
names(getModelInfo())
#mlr
listLearners("classif","twoclass")[c("class","package")]

#Algorithm 1: Native Bayes
#caret with 10-fold repeatedcv (with random split)
#if bootstrap, the sample size is 100% of train_DT.smote
#tune fL for nb to 1 for laplace smoothing but must tune all parameters, this is the cons of caret
#no need to use tuneLength=n (auto tuning searching from n values of each parameter) check default grid getModelInfo("nb")
#tune summaryFunction in trControl for full estimation (ACC/SENS/SPEC) after resampling.
#must omit NA in smote;must classProbs = TRUE (for AUC in twoClassSummary);
#most important: seperately input x and y
#unfortunately, tradeoff between sens and spec is observed
NB_caret_imbalance <- caret::train(x=subset(train.task.caret,select=-c(income_level)),y=train.task.caret$income_level,method="nb",na.action = na.omit,tuneLength=10,trControl=trainControl(method="repeatedcv",number=10,repeats=3,summaryFunction=twoClassSummary,classProbs = TRUE,allowParallel=TRUE))
NB_caret <- caret::train(x=subset(train_DT.smote,select=-c(income_level)),y=train_DT.smote$income_level,method="nb",na.action = na.omit,tuneLength=10,trControl=trainControl(method="repeatedcv",number=10,repeats=3,summaryFunction=twoClassSummary,classProbs = TRUE,allowParallel=TRUE))
NB_caret.rose <- caret::train(x=subset(train_DT.rose,select=-c(income_level)),y=train_DT.rose$income_level,method="nb",na.action = na.omit,tuneLength=10,trControl=trainControl(method="repeatedcv",number=10,repeats=3,summaryFunction=twoClassSummary,classProbs = TRUE,allowParallel=TRUE))
confusionMatrix(predict(NB_caret,test_DT),test_DT$income_level)
#mlr
#available models:https://mlr-org.github.io/mlr-tutorial/devel/html/integrated_learners/index.html
NB_mlr <- makeLearner("classif.naiveBayes",predict.type = "response")
#hyperparameters:laplace smoothing https://stats.stackexchange.com/questions/108797/in-naive-bayes-why-bother-with-laplacian-smoothing-when-we-have-unknown-words-i
#pros of mlr:can tune single parameter
#getParamSet("classif.xgboost") to check model hyperparameters
NB_mlr$par.vals <- list(laplace = 1)
#10-fold CV
#smote shows the best performance
folds <- makeResampleDesc("RepCV",rep=3)
NB_mlr_rcv_imbalance <- resample(NB_mlr,train.task.mlr,folds,measures = list(acc,tpr,tnr,fpr,fp,fn))#default measure for classif is mmce (mean misclassification error)
NB_mlr_rcv_under <- resample(NB_mlr,train.under,folds,measures = list(acc,tpr,tnr,fpr,fp,fn))#default measure for classif is mmce (mean misclassification error)
NB_mlr_rcv_over <- resample(NB_mlr,train.over,folds,measures = list(acc,tpr,tnr,fpr,fp,fn))#default measure for classif is mmce (mean misclassification error)
NB_mlr_rcv <- resample(NB_mlr,train.smote,folds,measures = list(acc,tpr,tnr,fpr,fp,fn))#default measure for classif is mmce (mean misclassification error)
NB_mlr_rcv_imbalance$aggr;NB_mlr_rcv$aggr;NB_mlr_rcv_under$aggr;NB_mlr_rcv_over$aggr
#Final training for mlr
#According to cross validation we choose smote only
#for pure training without cv is very fast
NB_mlr_model <- mlr::train(NB_mlr,train.smote)
NB_mlr_predict <- predict(NB_mlr_model,test.task.mlr)
NB_mlr_eva <- confusionMatrix(NB_mlr_predict$data$response,getTaskTargets(test.task.mlr))

###Algorithm 2: xgboost
#:1.parallel 2.grow trees to max_depth and prune backward
xgb_mlr <- makeLearner("classif.xgboost",predict.type = "response")
xgb_mlr$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 150,
  print.every.n=50
)
#for xgb we need to do grid search via cv
#getParamSet("classif.xgboost") to check model hyperparameters
#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
xg_hparam <- makeParamSet(
  makeIntegerParam("max_depth",lower=3,upper=10),#tree depth typically [3,10]
  makeNumericParam("lambda",lower = 0.05,upper=0.5),#regularization
  makeNumericParam("eta",0.01,0.5),#learning rate typically[0.01,0.2]
  makeNumericParam("subsample",0.5,1),#random sample ratio for training the tree
  makeNumericParam("min_child_weight",2,10),#controls the minimal sum weight of a leaf node (is sum(w)<MCW,
  #tree stops growing. So model will not be overfitted due to sparse observations in a subset)
  makeNumericParam("colsample_bytree",0.5,0.8)#random feature ratio for training the tree
)
#search function for the hyperparame grid:random search here
#grid search: go through all combinations with predefined searching resolution 
#random: random sample param sets 
#random>grid:some paramters are not influencing so better skip for speed;no limitation from searching resolution
tuneControl <- makeTuneControlRandom(maxit=5)
#change factor to dummy because xgboost doesn't support factor
#dummy:n level factor -> n variables with 1/0 
train.smote.dummy <- createDummyFeatures(train.smote)
#tune the parameters via cv
xgb_tune <- tuneParams(learner=xgb_mlr,task = train.smote.dummy,resampling = folds,
                       measures = list(acc,tpr,tnr,fpr,fp,fn),
                       par.set = xg_hparam,
                       control = tuneControl
)
#optimal xgb
xgb_opt <- setHyperPars(learner = xgb_mlr,par.vals = xgb_tune$x)
#train
xgb_model <- mlr::train(xgb_opt,train.smote.dummy)
xgb_model_imbalance <- mlr::train(xgb_opt,createDummyFeatures(train.task.mlr))
#predict:very poor sensitivity to "high"
#what happened?
test.task.mlr.dummy <- createDummyFeatures(test.task.mlr)
xgb_predict <- predict(xgb_model,test.task.mlr.dummy)
confusionMatrix(xgb_predict$data$response,test$income_level)
xgb_predict_imbalance <- predict(xgb_model_imbalance,test.task.mlr.dummy)
confusionMatrix(xgb_predict_imbalance$data$response,test$income_level)
library(ROSE)
roc.curve(NB_mlr_predict$data$response,test$income_level,plotit = T)
roc.curve(xgb_predict$data$response,test$income_level,plotit = T)
roc.curve(xgb_predict_imbalance$data$response,test$income_level,plotit = T)


#caret xgboost
#for each parameter need to specify all values (not only min and max)
#customized grid would disable tuneLength
#tuneLength only works for: 1. non-customized grid 2. random search 
xg_hparam <- expand.grid(
  nrounds = c(150),
  max_depth=seq(3,10,3),#tree depth typically [3,10]
  gamma=c(0.01),#regularization
  eta=seq(0.01,0.2,0.1),#learning rate typically[0.01,0.2]
  subsample=seq(0.5,1,0.2),#random sample ratio for training the tree
  min_child_weight=seq(2,10,3),#controls the minimal sum weight of a leaf node (is sum(w)<MCW,
  #tree stops growing. So model will not be overfitted due to sparse observations in a subset)
  colsample_bytree=c(0.5,0.8)#random feature ratio for training the tree
)
#may get error since XGBoost won't work with categorical (factor) variables
#so use one-hot dummy variables, in caret this is aotumated if we pass in "y~x" to train
#verbose=TRUE: +fold is traning -fold is teting
xgb_caret <- caret::train(income_level~.,data=train_DT.smote,method="xgbTree",na.action = na.omit,tuneLength=10,trControl=trainControl(method="repeatedcv",number=10,repeats=3,summaryFunction=twoClassSummary,classProbs = TRUE,allowParallel=TRUE,verboseIter=TRUE,search = "random"))
#here must omit na using na.omit since we ignor na recors in training
#the performance of xgb highly depends on hyperparam
xgb_caret_predict <- predict(xgb_caret,na.omit(test_DT))
confusionMatrix(xgb_caret_predict,na.omit(test_DT)$income_level)

#RFE
#try feature elimination:rank(S) and go through S_i to S_s
#sizes: the sizes for each S_i
#linear regression (in the object lmFuncs), random forests (rfFuncs), naive Bayes (nbFuncs), bagged trees (treebagFuncs)
train_DT.smote.rfe <- rfe(na.omit(train_DT.smote)%>%select(-income_level),na.omit(train_DT.smote)$income_level,sizes=c(1:5),rfeControl=rfeControl(functions = treebagFuncs))
#choose top 10
#for character names use one_of
train_DT.smote.top10 <- train_DT.smote %>% select(one_of(train_DT.smote.rfe$optVariables[1:10],"income_level"))
#retrain xgb
xgb_caret <- caret::train(income_level~.,data=train_DT.smote.top10,method="xgbTree",na.action = na.omit,tuneLength=10,trControl=trainControl(method="repeatedcv",number=10,repeats=3,summaryFunction=twoClassSummary,classProbs = TRUE,allowParallel=TRUE,verboseIter=TRUE,search = "random"))

xgb_caret_predict <- predict(xgb_caret,na.omit(test_DT),type = "prob") %>% mutate(pred=ifelse(low>0.82,"low","high"))
xgb_caret_predict$pred <- as.factor(xgb_caret_predict$pred)
confusionMatrix(xgb_caret_predict$pred,na.omit(test_DT)$income_level)
roc.curve(xgb_caret_predict$pred,na.omit(test_DT)$income_level)
