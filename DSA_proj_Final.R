######################
# load the dataset
######################
library(readr) #load readr package
coldcall <- read_csv("C:/Users/Dehan/Desktop/IMPT/4. SMU school/4, 2019 SEM 2 MODS/3. DSA211/proj/carInsurance_train.csv", 
                     col_types = cols(CallEnd = col_time(format = "%H:%M:%S"), 
                                      CallStart = col_time(format = "%H:%M:%S"), 
                                      CarInsurance = col_factor(levels = c()), 
                                      CarLoan = col_factor(levels = c()), 
                                      Communication = col_factor(levels = c()), 
                                      Default = col_factor(levels = c()), 
                                      Education = col_factor(levels = c()), 
                                      HHInsurance = col_factor(levels = c()), 
                                      Id = col_skip(), 
                                      Job = col_factor(levels = c()), 
                                      LastContactDay = col_skip(), 
                                      LastContactMonth = col_skip(), 
                                      Marital = col_factor(levels = c()), 
                                      Outcome = col_factor(levels = c())))
str(coldcall) #exploratory data analysis
################################
# Data Cleaning and Preparation
################################
coldcall<-na.omit(coldcall) #removing rows with NA values
library(dplyr) #load dplyr package
library(lubridate) #load lubridate package
coldcall<-coldcall%>%
  mutate(CallDuration=time_length(interval(coldcall$CallStart,coldcall$CallEnd),unit = 'minute'))%>%
  select(-CallStart,-CallEnd) #Removing CallStart and CallEnd which are not useful since we have created the column CallDuration
str(coldcall)
summary(coldcall) #examining cleaned data set
##############################
# Split into training and test set
##############################
RNGkind(sample.kind = "Rounding") #standardising RNG process
set.seed(1) #set random seed for consistency in results
train<-sample(1:nrow(coldcall),round(0.8*nrow(coldcall))) #create train data set using 80% of original data
test<-(-train) #create test dataset using 20% of original data

#############################
# Logistic Regression 
#############################
m1<-glm(CarInsurance~.,data = coldcall[train,],family = binomial) #fitting the logistic model
summary(m1) #looking at the fit of the model
pvalue1 <- 1-pchisq(630.98, 699)
pvalue1 #hypo test for whether the model is a good fit at a 5% level of significance
coef(m1)
coldcall$Entrepreneur<-as.factor(ifelse(coldcall$Job=='entrepreneur','1','0')) #converting Entrepeneur categorical data into factors and adding the new column to the dataset
coldcall$PrevSucc<-as.factor(ifelse(coldcall$Outcome=='success','1','0')) #converting PrevSuccess categorical data into factors and adding the new column to the dataset
new_coldcall<-as.data.frame(coldcall) #creating a new data frame with which is a copy of the cleaned dataset for ease of reference
m11<-glm(CarInsurance~Entrepreneur+HHInsurance+CarLoan+NoOfContacts+PrevSucc+CallDuration,data = new_coldcall[train,],family = binomial) #running only the 5% significant models
summary(m11) #examining m11 fit and coeffs
pvalue2 <- 1-pchisq(655.61, 719)
pvalue2 #hypo test for whether the model is a good fit at a 5% level of significance
coef(m11)

# Creating the m1 confusion matrix
# 0 = Cold Call not successful, 1 = Cold Call successful
m1.prob<-predict(m1,coldcall[test,],type = 'response')
m1.pred<-rep('0',nrow(coldcall[test,]))
m1.pred[m1.prob>0.5]<-'1'
table(coldcall[test,]$CarInsurance)
table(m1.pred,coldcall[test,]$CarInsurance) #m1 confusion matrix

#creating the m11 confusion matrix
# 0 = Cold Call not successful, 1 = Cold Call successful
m11.prob<-predict(m11,new_coldcall[test,],type = 'response')
m11.pred<-rep('0',nrow(new_coldcall[test,]))
m11.pred[m11.prob>0.5]<-'1'
table(m11.pred,new_coldcall[test,]$CarInsurance) #m11 confusion matrix

############################
#Best Subset 
############################
library(leaps) #loading the leaps package
m2<-regsubsets(CarInsurance~.,coldcall[train,],nvmax=14) #forming the model
m2.summary<-summary(m2) #examinging model m2 fit and coefficients
plot(m2.summary$bic,main = 'BIC plot',xlab = 'number of predictors',ylab = 'BIC') #plot number of predictors vs BIC
b<-which.min(m2.summary$bic) #find the lowest BIC generated
coef(m2,b) #find model m2 coeffs using best number of predictors
m22<-glm(CarInsurance~HHInsurance+PrevSucc+CallDuration, data = new_coldcall[train,],family = binomial) #fitting log model m22
summary(m22) #looking at model m22 fit and coeffcients

#creating a confusion matrix for m22
m22.prob<-predict(m22,new_coldcall[test,],type = 'response')
m22.pred<-rep('0',nrow(new_coldcall[test,]))
m22.pred[m22.prob>0.5]<-'1'
table(m22.pred,new_coldcall[test,]$CarInsurance) #confusion matrix created

#############################################
#Ridge and Lasso 
############################################
library(glmnet) #load glmnet package
x<-model.matrix(CarInsurance~.,coldcall[train,])[,-1]
y<-coldcall[train,]$CarInsurance
new.x<-model.matrix(CarInsurance~.,coldcall[test,])[,-1]
new.y<-coldcall[test,]$CarInsurance

ridge<-glmnet(x,y,alpha = 0, nlambda = 100,family = 'binomial')
cv<-cv.glmnet(x,y,alpha=0, family = 'binomial')
plot(cv)
bestlamb<-cv$lambda.min
bestlamb
coef(ridge,bestlamb)

ridge.pred<-predict(ridge,s=bestlamb, newx = new.x,type = 'response')
prediction<-rep('0',length(new.y))
prediction[ridge.pred>0.5]<-'1'
table(prediction,new.y)

lasso<-glmnet(x,y,alpha = 1, nlambda = 100,family = 'binomial')
cv2<-cv.glmnet(x,y,alpha=1, family = 'binomial')
plot(cv2)
bestlamb2<-cv2$lambda.min
bestlamb2
coef(lasso,bestlamb2)

lasso.pred<-predict(lasso,s=bestlamb2, newx = new.x,type = 'response')
prediction2<-rep('0',length(new.y))
prediction2[lasso.pred>0.5]<-'1'
table(prediction2,new.y)

##############################
#Decision Tree 
##############################
library(tree)
mtree<-tree(CarInsurance~.,coldcall,subset = train)
summary(mtree)
mtree
plot(mtree)
title(main = 'Success of Cold Call')
text(mtree,pretty = 0)

prune_mtree<-cv.tree(mtree,FUN = prune.misclass)
plot(prune_mtree$size,prune_mtree$dev,type = 'b',main = 'pruning: size versus deviance',xlab = 'number of nodes',ylab = 'deviance')
nn<-prune_mtree$size[which.min(prune_mtree$dev)]
nn
prune_model<-prune.misclass(mtree,best = nn)
plot(prune_model)
title(main = 'Success of Cold Call')
text(prune_model,pretty = 0)

tree_pred<-predict(mtree,coldcall[test,],type = 'class')
table(tree_pred,coldcall[test,]$CarInsurance)

prune_pred<-predict(prune_model,coldcall[test,],type = 'class')
table(prune_pred,coldcall[test,]$CarInsurance)

