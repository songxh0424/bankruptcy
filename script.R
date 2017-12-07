setwd("/Users/Carl/Google Drive/2017 winter/503/project")
library(foreign)
library(tidyr)
library(dplyr)
library(ggplot2)
library(MASS)
library(leaps)
library(class)
library(randomForest)

b = list()
for (i in 1:5) {
  b[[i]] = read.arff(paste("bank/", i, "year.arff", sep = ""))
  write.csv(b[[i]], file = paste("bank/", i, "year.csv", sep = ""))
}


bank = read.arff("bank/5year.arff")
summary(bank)
bank_lda = lda(data = bank, class ~ ., na.action = na.omit)
temp = na.omit(bank)

gesture = read.csv("gesture_phase_dataset/")

train = sample(1:nrow(temp), size = 2000, replace = FALSE)
temp_train = temp[train, ]
temp_test = temp[-train, ]

temp_pca = princomp(temp_train[, -65], cor = TRUE)
A = cumsum(temp_pca$sdev^2) / sum(temp_pca$sdev^2)
ggplot(data.frame(), aes(1:64, A)) + geom_line()
ggplot(data.frame(), aes(1:64, temp_pca$sdev)) + geom_line()

train_pca = data.frame(temp_pca$scores[, 1:20], class = temp_train$class)
test_pca = as.matrix(temp_test[, -65]) %*% temp_pca$loadings[, 1:20]
test_pca = data.frame(test_pca, class = temp_test$class)
train_knn = knn(train_pca[, -21], test_pca[, -21], train_pca[, 21], k = 5)
mean(train_knn != test_pca$class)
table(train_knn, test_pca$class)
train_knn

train_rf = randomForest(class ~ ., data = temp_train)
pred = predict(train_rf, temp_test)
table(pred, temp_test$class)

bank_std = scale(bank[, -65])
bank_std = data.frame(bank_std, class = bank$class)
long = gather(data = bank_std, key = attr, value = value, -class)
long = na.omit(long)
ggplot(long, aes(attr, value, fill = class)) + geom_boxplot() + coord_flip()


