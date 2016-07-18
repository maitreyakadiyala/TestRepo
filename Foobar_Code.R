#Loading Libraries#
install.packages("Hmisc")
library(Hmisc)
install.packages("data.table")
library(data.table)
install.packages("dplyr")
library(dplyr)
install.packages("missForest")
library(missForest)
install.packages("sqldf")
library(sqldf)
install.packages("VIM")
library(VIM)
install.packages("mice")
library(mice)

#Working Directories
setwd("C:/Users/1320128/Desktop/Foobar-Project")
#Loading Data
foobar_v1 <- fread(
  input = "widgets.csv",
  sep = "",
  showProgress = getOption("datatable.showProgress")
)
str(foobar_v1)
describe(foobar_v1)
foobar_v1
##############################
#imputing missing values in R#
##############################
mydata <-
  read.table(
    "C:/Users/1320128/Desktop/Foobar-Project/widgets.csv",
    header = TRUE,
    sep = " "
  )
foobar_v1_imp <- kNN(
  mydata,
  variable = c("style"),
  dist_var = c("construction", "size", "weight", "height", "quality", "price"),
  k = 5,
  imp_suffix = "_imp"
)

mydata_new <-
  sqldf("select * from foobar_v1_imp where price > 0  and height >0")

install.packages("caret")
library(caret)

names(getModelInfo())
set.seed(1234)
sample_data <- mydata_new[sample(nrow(mydata_new)), ]
split <- floor(nrow(mydata_new) / 3)
train_mydata <- mydata_new[0:split, ]
validation_mydate <- mydata_new[(split + 1):(split * 2), ]
library(MASS)
barplot(table(mydata_new$style),cex.names = 3)

testing_mydata <- mydata_new[(split * 2 + 1):nrow(mydata_new), ]
labelname <- 'price'
predictors <- names(train_mydata)[names(train_mydata) != 'price']
mycontrol <-
  trainControl(
    method = "cv",
    number = 3,
    repeats = 1,
    returnResamp = "none"
  )

set.seed(1234)

#Model 1
model_lm_1 <- lm(formula = price~., data = train_mydata) 
summary(model_lm_1)

par(mfrow = c(1,1))
png('R Visualization/LinearModel01%03d.png', width=2000, height=2000, res=300)
plot(model.lm.1, 1:6, ask = FALSE)
dev.off()

model.lm.1.varImp <- varImp(model.lm.1)
model.lm.1.varImp

#Predict via Model 1
predict.lm.1<-predict(model.lm.1,data.test) 
predict.lm.1.modelvalues<-data.frame(obs = data.test$price, pred=predict.lm.1)
defaultSummary(predict.lm.1.modelvalues)

#Model 2
model.lm.2 <- lm(formula = price ~ style + quality + height + size + construction, data = data.train)
summary(model.lm.2)

par(mfrow = c(1,1))
png('R Visualization/LinearModel02%03d.png', width=2000, height=2000, res=300)
plot(model.lm.1, 1:6, ask = FALSE)
dev.off()

model.lm.2.varImp <- varImp(model.lm.2)
model.lm.2.varImp

#Predict via Model 2
predict.lm.2 <- predict(model.lm.2, data.test)  
predict.lm.2.modelvalues<-data.frame(obs = data.test$price, pred=predict.lm.2)
defaultSummary(predict.lm.2.modelvalues)

#Comparison of Original Price And Predicted Price Values
predict(model.lm.2, data.test, se.fit = TRUE)
predict.w.plim <- predict(model.lm.2, data.test, interval = "prediction")

png('R Visualization/LinearModel.PredVsObs.png', width=2000, height=2000, res=300)
matplot(data.test$price,predict.w.plim,lty = c(1,2,2,3,3), type = "l", ylab = "Predicted Price", xlab = "Original Price")
dev.off()

#Prepare Training Schema
control <- trainControl(method="repeatedcv", number=10, repeats=3)

#Caret Linear Model Train
set.seed(28262)
model.caret.Lm <- train(price ~ style + quality + height + size + construction, data=data.train, method="lm", trControl=control)
model.caret.Lm

#Caret Linear Model Predict
predict.caret.Lm <- predict(model.caret.Lm, newdata = data.test)
predict.caret.Lm.modelvalues <- data.frame(obs = data.test$price, pred=predict.caret.Lm)
defaultSummary(predict.caret.Lm.modelvalues)

#Caret GBM Model
set.seed(28262)
model.caret.Gbm <- train(price ~ style + quality + height + size + construction, data=data.train, method="gbm", 
                         trControl=control, verbose=FALSE)
model.caret.Gbm

#Caret GBM Model Predict
predict.caret.Gbm <- predict(model.caret.Gbm, newdata = data.test)
predict.caret.Gbm.modelvalues <- data.frame(obs = data.test$price, pred=predict.caret.Gbm)
defaultSummary(predict.caret.Gbm.modelvalues)

#Collecting Resamples
set.seed(28262)
model.caret.results <- resamples(list(LM=model.caret.Lm, GBM=model.caret.Gbm))

#Summary
summary(model.caret.results)

#BoxPlot Of Results
png('R Visualization/CaretModelsA.PredVsObs.png', width=2000, height=2000, res=300)
bwplot(model.caret.results)
dev.off()

#DotPlot Of Results
png('R Visualization/CaretModelsB.PredVsObs.png', width=2000, height=2000, res=300)
dotplot(model.caret.results)
dev.off()


#Print Results
print(model.caret.results$values)

#======================================================================
#RANDOMFOREST MODEL (RANDOMFOREST LIBRARY)
#======================================================================
library(rattle)
write.csv(data.train, "train.csv")
write.csv(data.test, "test.csv")
rattle()

#updated by Venkata
#step 4 comment