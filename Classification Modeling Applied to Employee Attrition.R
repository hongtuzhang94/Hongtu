library(class)
library(caret)
library(ggplot2)
library(e1071)
library(gmodels)
library(forecast)
library(adabag)
library(rpart) 
library(caret)

attrition <- read.csv("EmployeeAttrition.csv")
attrition <- attrition[,c(1:3,6,7,11,12,17:19,23:25,29,33,34)]

ggplot(attrition, aes(x = factor(Attrition))) + 
  geom_bar(fill = 'lightblue')

ggplot(attrition, aes(x = factor(Attrition), y = Age)) + 
  geom_boxplot() + 
  scale_fill_brewer(palette='Dark2') +
  theme(panel.background = element_rect(fill='lightblue', colour='lightblue'))

ggplot(attrition, aes(x = factor(Attrition), y = MonthlyIncome)) + 
  geom_point()

ggplot(attrition, aes(x = Age, y = MonthlyIncome, col = factor(Attrition))) + 
  geom_point()

# rewrite predictors
attrition$BusinessTravel <- ifelse(attrition$BusinessTravel == "Travel_Frequently", 1, 0)

attrition$Gender <- ifelse(attrition$Gender == "Male", 1, 0)

attrition$MaritalStatus <-ifelse(attrition$MaritalStatus == "Single", 1, 
                                 ifelse(attrition$MaritalStatus == "Married", 2, 3))

attrition$Attrition <- ifelse(attrition$Attrition == "Yes", 1, 0)

attrition$OverTime <- ifelse(attrition$OverTime == "Yes", 1, 0)

nomalized <- scale(attrition[,-2])
attrition.norm <- cbind(nomalized,attrition$Attrition)
colnames(attrition.norm)[16] <- "Attrition"
         
set.seed(1)
train.index <- sample(c(1:dim(attrition.norm)[1]), dim(attrition.norm)[1]*0.6)
train.df <- attrition.norm[train.index, ]
train.df <- as.data.frame(train.df)
valid.df <- attrition.norm[-train.index, ]
valid.df <- as.data.frame(valid.df)

knn1 <- knn(train.df[,1:15], valid.df[,1:15], cl = train.df[,16] , 
            k= 1, prob = FALSE)
knn2 <- knn(train.df[,1:15], valid.df[,1:15], cl = train.df[,16] , 
            k= 2, prob = FALSE)
knn3 <- knn(train.df[,1:15], valid.df[,1:15], cl = train.df[,16] , 
            k= 3, prob = FALSE)
knn4 <- knn(train.df[,1:15], valid.df[,1:15], cl = train.df[,16] , 
            k= 4, prob = FALSE)
knn5 <- knn(train.df[,1:15], valid.df[,1:15], cl = train.df[,16] , 
            k= 5, prob = FALSE)
knn6 <- knn(train.df[,1:15], valid.df[,1:15], cl = train.df[,16] , 
            k= 6, prob = FALSE)
knn7 <- knn(train.df[,1:15], valid.df[,1:15], cl = train.df[,16] , 
            k= 7, prob = FALSE)

table(knn1, valid.df[,16])
table(knn2, valid.df[,16])
table(knn3, valid.df[,16])
table(knn4, valid.df[,16])
table(knn5, valid.df[,16])
table(knn6, valid.df[,16])
table(knn7, valid.df[,16])

# choose k = 7

attrition.knn <- knnreg(Attrition~.,train.df, k=7)
attrition.knn.pred <- predict(attrition.knn, valid.df)

attrition.lift <- data.frame(case = c(1:dim(valid.df)[1]),
                       knnreg = cumsum(valid.df$Attrition[order(attrition.knn.pred, decreasing = T)]),
                       baseline = c(1:dim(valid.df)[1])*mean(valid.df$Attrition))

ggplot(attrition.lift, aes(x = case)) + 
  geom_line(aes(y = knnreg), color = "blue") +
  geom_line(aes(y=baseline), color = "red", linetype = "dashed") + 
  labs(x = "# of cases", y = "Cumulative Expenses")

# naive bayes
attrition1 <- attrition
# transform to factors
attrition1$Age <- factor(round(attrition1$Age/10))
attrition1$DistanceFromHome <- factor(round(attrition1$DistanceFromHome/5))
attrition1$MonthlyIncome <- factor(round(attrition1$MonthlyIncome/3000))
attrition1$PercentSalaryHike <- factor(round(attrition1$PercentSalaryHike/10))
attrition1$TotalWorkingYears <- factor(round(attrition1$TotalWorkingYears/5))
attrition1$YearsInCurrentRole <- factor(round(attrition1$YearsInCurrentRole/3))
attrition1$YearsSinceLastPromotion <- factor(round(attrition1$YearsSinceLastPromotion/3))
attrition1$Education <- as.factor(attrition1$Education)
attrition1$EnvironmentSatisfaction <- as.factor(attrition1$EnvironmentSatisfaction)
attrition1$Attrition <- as.factor(attrition1$Attrition)
attrition1$BusinessTravel <- as.factor(attrition1$BusinessTravel)
attrition1$Gender <- as.factor(attrition1$Gender)
attrition1$MaritalStatus <- as.factor(attrition1$MaritalStatus)
attrition1$JobSatisfaction <- as.factor(attrition1$JobSatisfaction)
attrition1$OverTime <- as.factor(attrition1$OverTime)
attrition1$PerformanceRating <- as.factor(attrition1$PerformanceRating)

train.df1 <- attrition1[train.index, ]
train.df1 <- as.data.frame(train.df1)
valid.df1 <- attrition1[-train.index, ]
valid.df1 <- as.data.frame(valid.df1)

attrition.nb <- naiveBayes(Attrition ~ ., data = train.df1)
attrition.nb 

# predict prob
pred.prob <- predict(attrition.nb, newdata = valid.df1, type = "raw")
# predict class in validation
pred.class.v <- predict(attrition.nb, newdata = valid.df1)
# predict class in validation
pred.class.t <- predict(attrition.nb, newdata = train.df1)

CrossTable(x = valid.df1$Attrition, y = pred.class.v)

df <- data.frame(actual = valid.df1$Attrition, 
                 predicted = pred.class.v, pred.prob)

# confusion matrix
table(pred.class.t, train.df1$Attrition)
table(pred.class.v, valid.df1$Attrition)

# CART
library(rpart)
library(rpart.plot)
attrition <- read.csv("EmployeeAttrition.csv")
attrition <- attrition[,c(1:3,6,7,11,12,17:19,23:25,29,33,34)]

set.seed(1)

train.index2 <- sample(c(1:dim(attrition)[1]), 
                       dim(attrition)[1]*0.5)
train.df2 <- attrition[train.index2, ]
train.df2 <- as.data.frame(train.df2)
except.train2 <- attrition[-train.index2, ]
valid.index2 <- sample(c(1:dim(except.train2)[1]), 
                       dim(except.train2)[1]*0.6)
valid.df2 <- except.train2[valid.index2, ]
valid.df2 <- as.data.frame(valid.df2)
test.df2 <- except.train2[-valid.index2, ]
test.df2 <- as.data.frame(test.df2)

# default tree
attrition.ct <- rpart(Attrition ~., data = train.df2, method = "class")
prp(attrition.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)

# full grown tree
deeper.ct <- rpart(Attrition ~., data = train.df2, 
                   method = "class", cp = 0.00001, minsplit = 1)
# count number of leaves
length(deeper.ct$frame$var[deeper.ct$frame$var == "<leaf>"])
# plot tree
prp(deeper.ct, type = 1, extra = 1, under = TRUE, 
    split.font = 1, varlen = -10,
    box.col=ifelse(deeper.ct$frame$var == "<leaf>", 'gray', 'white'))

# use printcp() to print the table. 
printcp(deeper.ct)

deeper.pred <- predict(deeper.ct, valid.df2, type = 'class')
table(deeper.pred, valid.df2$Attrition)

# prune by lower cp
pruned.ct <- prune(deeper.ct, cp = 0.0166667)
length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10) 
pruned.pred <- predict(pruned.ct, test.df2, type = 'class')
table(pruned.pred, test.df2$Attrition)

# randomforest
library(randomForest)
rf <- randomForest(as.factor(Attrition) ~ ., data = train.df2, 
                   ntree = 500, mtry = 4, 
                   nodesize = 5, importance = TRUE)  

# variable importance plot
varImpPlot(rf, type = 1)

# confusion matrix
rf.pred <- predict(rf, valid.df2, type = "class")
table(rf.pred, valid.df2$Attrition)

# boosting
set.seed(1)
train.df2$Attrition <- as.factor(train.df2$Attrition)
boost <- boosting(Attrition ~ ., data = train.df2)
pred <- predict(boost, valid.df2)
table(pred$class, valid.df2$Attrition)

# logistic
# use glm() (general linear model) with family = "binomial" to fit a logistic regression
set.seed(1)
train.index3 <- sample(c(1:dim(attrition)[1]), 
                       dim(attrition)[1]*0.6)
train.df3 <- attrition[train.index3, ]
train.df3 <- as.data.frame(train.df3)
valid.df3 <- attrition[-train.index3, ]
valid.df3 <- as.data.frame(valid.df3)

logit.reg <- glm(Attrition ~ ., data = train.df3, family = "binomial") 
options(scipen=999)
summary(logit.reg)
# use predict() with type = "response" to compute predicted probabilities. 
logit.reg.pred <- predict(logit.reg, valid.df3[,-2], type = "response")

# first 5 actual and predicted records
data.frame(actual = valid.df3$Attrition[1:5], 
           predicted = logit.reg.pred[1:5])

logit.reg.pred1 <- ifelse(logit.reg.pred > 0.5, 1, 0)
table(valid.df3$Attrition, logit.reg.pred1)


library(gains)
gain <- gains(as.numeric(valid.df3$Attrition), logit.reg.pred, groups=10)

# plot lift chart
plot(c(0,gain$cume.pct.of.total*sum(as.numeric(valid.df3$Attrition)))~c(0,gain$cume.obs), 
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(as.numeric(valid.df3$Attrition)))~c(0, dim(valid.df3)[1]), 
      lty=2)

