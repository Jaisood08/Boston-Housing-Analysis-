#--------------------------------------------------------------
# Step 1: Importing Data
#--------------------------------------------------------------

library(MASS)
data(Boston)
View(head(Boston))
dim(Boston)

summary(Boston)

#--------------------------------------------------------------
# Step 2: Knowing Data
#--------------------------------------------------------------

library(psych)
describe(Boston)

#--------------------------------------------------------------
# Step 3: Plotting Data
#--------------------------------------------------------------

library("PerformanceAnalytics")
correlations = cor(Boston)
chart.Correlation(correlations, histogram=TRUE, pch=20,cex = 5)

#--------------------------------------------------------------
# Step 4: Different Correlation Plots
#--------------------------------------------------------------

library(corrplot)
library(corrgram)
correlations = cor(Boston)
corrgram(correlations)
corrplot(correlations,order="hclust",method="square",
         tl.cex = 1,cl.cex = 0.6)

#--------------------------------------------------------------
# Step 5: Distribution Plots
#--------------------------------------------------------------

par(mfrow=c(3, 3))
colnames <- dimnames(Boston)[[2]]
colnames
for (i in 1:14) 
{
  hist(Boston[,i], main=colnames[i], probability=TRUE, col="blue", 
       border="white")
  d <- density(Boston[,i])
  lines(d, col="red")
}
par(mfrow=c(1, 1))
#--------------------------------------------------------------
# Step 6: Preparing Data For Model
#--------------------------------------------------------------

colnames <- dimnames(Boston)[[2]]
colnames
Target = colnames[length(colnames)]
Target
Features = colnames[0:length(colnames)]
cat(Features)

acceptableError = 3

index <- sample(nrow(Boston),nrow(Boston)*0.70) #70-30 split
Boston.train <- Boston[index,]
Boston.test <- Boston[-index,]
cat("Training Set : ",dim(Boston.train))
cat("Testing Set : ",dim(Boston.test))

#--------------------------------------------------------------
#                 7. Generating A linear model
#--------------------------------------------------------------

#--------------------------------------------------------------
# Step 7.1: Cross Validation
#--------------------------------------------------------------
library(caret)
CV <- trainControl(method = "repeatedcv",
                   number = 10,
                   repeats = 5,
                   verboseIter = T)

model <- train(medv~.,
               Boston.train,
               method = "lm",
               trControl = CV)

model$results
summary(model)
#--------------------------------------------------------------
# Step 7.2: Prediction
#--------------------------------------------------------------
Actual = as.double(unlist(Boston.test[Target]))
Predicted = predict(model,Boston.test)
#--------------------------------------------------------------
# Step 7.3: Model Evaluation
#--------------------------------------------------------------
# Correlation
r <- round(cor(Actual,Predicted),2)

# RSquare
R <- round(r * r,2) 

# MAE (Mean Absolute)
mae <- round(mean(abs(Actual-Predicted)),2)

# Accuracy
accuracy <- round(mean(abs(Actual-Predicted) <=acceptableError),4)*100

cat("Correlation : ",r,"\n",
    "RSquare : ",R,"\n",
    "MAE : ",mae,"\n",
    "Accuracy : ",accuracy)

#--------------------------------------------------------------
#                 8. Ridge Regularization 
#--------------------------------------------------------------

ridge <- train(medv~.,
               Boston.train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha=0,
                                      lambda = seq(0.0001,1,length=5)),
               trControl = CV)

plot(ridge)
plot(varImp(ridge,scale = T))

#--------------------------------------------------------------
# Step 8.1: Prediction
#--------------------------------------------------------------
Actual = as.double(unlist(Boston.test[Target]))
Predicted = predict(ridge,Boston.test)
#--------------------------------------------------------------
# Step 8.2: Model Evaluation
#--------------------------------------------------------------
# Correlation
r <- round(cor(Actual,Predicted),2)

# RSquare
R <- round(r * r,2) 

# MAE (Mean Absolute)
mae <- round(mean(abs(Actual-Predicted)),2)

# Accuracy
accuracy <- round(mean(abs(Actual-Predicted) <=acceptableError),4)*100

cat("Correlation : ",r,"\n",
    "RSquare : ",R,"\n",
    "MAE : ",mae,"\n",
    "Accuracy : ",accuracy)

#--------------------------------------------------------------
#                 9. Lasso Regularization 
#--------------------------------------------------------------


lasso <- train(medv~.,
               Boston.train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha=1,
                                      lambda = seq(0.0001,1,length=5)),
               trControl = CV)
plot(lasso)
plot(varImp(lasso,scale = T))

#--------------------------------------------------------------
# Step 9.1: Prediction
#--------------------------------------------------------------
Actual = as.double(unlist(Boston.test[Target]))
Predicted = predict(lasso,Boston.test)
#--------------------------------------------------------------
# Step 9.2: Model Evaluation
#--------------------------------------------------------------
# Correlation
r <- round(cor(Actual,Predicted),2)

# RSquare
R <- round(r * r,2) 

# MAE (Mean Absolute)
mae <- round(mean(abs(Actual-Predicted)),2)


# Accuracy
accuracy <- round(mean(abs(Actual-Predicted) <=acceptableError),4)*100

cat("Correlation : ",r,"\n",
    "RSquare : ",R,"\n",
    "MAE : ",mae,"\n",
    "Accuracy : ",accuracy)

#--------------------------------------------------------------
#                 10. Elastic Net Regularization 
#--------------------------------------------------------------

EN <- train(medv~.,
               Boston.train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha=seq(0,1,length=10),
                                      lambda = seq(0.0001,1,length=5)),
               trControl = CV)
plot(EN)
plot(varImp(EN,scale = T))

#--------------------------------------------------------------
# Step 10.1: Prediction
#--------------------------------------------------------------
Actual = as.double(unlist(Boston.test[Target]))
Predicted = predict(EN,Boston.test)
#--------------------------------------------------------------
# Step 10.2: Model Evaluation
#--------------------------------------------------------------
#  Correlation
r <- round(cor(Actual,Predicted),2)

#  RSquare
R <- round(r * r,2) 

# MAE (Mean Absolute)
mae <- round(mean(abs(Actual-Predicted)),2)

# Accuracy
accuracy <- round(mean(abs(Actual-Predicted) <=acceptableError),4)*100

cat("Correlation : ",r,"\n",
    "RSquare : ",R,"\n",
    "MAE : ",mae,"\n",
    "Accuracy : ",accuracy)


#--------------------------------------------------------------
#                 11. Compare All Models
#--------------------------------------------------------------


model_list = list(LinearModel=model,
                  Ridge = ridge,
                  Lasso = lasso,
                  ElasticNet = EN)


Result <- resamples(model_list)

summary(Result)               



#--------------------------------------------------------------
#                 12. Result
#--------------------------------------------------------------

plot(varImp(ridge,scale = F))
