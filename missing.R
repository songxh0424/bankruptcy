setwd("/Users/Carl/Google Drive/2017 winter/503/project")
library(foreign) # read.arff
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(MASS)
library(class)
library(randomForest)
library(mice) # multiple imputation
library(missForest) # impute with RF
library(adabag) # boosting
library(caret) # train models
library(doMC) # parallel
library(knitr)
library(sparsediscrim)

load("project.RData")

bank = read.arff("bank/5year.arff")
# remove Attr37, 43% NA
bank = bank[, -37]

summary(bank)
colSums(is.na(bank)) / 5910

# multiple imputation
bank_imputed = mice(bank[, -64], method = "pmm", seed = 1234)

# randomForest imputation
bank_imputed2 = missForest(bank[, -64])

# orginal data with na imputed
dat = bank_imputed2$ximp
dat$class = bank$class
dat = mutate(dat, class = factor(ifelse(class == 1, "Yes", "No"), levels = c("Yes", "No")))
set.seed(1234)
ratio = 0.7
index = sample(1:nrow(dat), floor(ratio * nrow(dat)))
train = dat[index, ]
test = dat[-index, ]
# standardized
dat.std = dat
dat.std[, -64] = scale(dat[, -64])
train.std = dat.std[index, ]
test.std = dat.std[-index, ]
# with number of na as a predictor
dat.na = dat
dat.na$numNA = rowSums(is.na(bank))
train.na = dat.na[index, ]
test.na = dat.na[-index, ]
# with the first 20 PCs
pcacor = princomp(train[, -64], cor = TRUE)
train.pca = as.data.frame(pcacor$scores[,1:20])
train.pca$class = train$class
names(train.pca)[1:20] = paste("Comp.", 1:20, sep = "")
test.pca = as.data.frame(as.matrix(test[,-64] - outer(rep(1,nrow(test)),colMeans(train[,-64]))) %*% as.matrix(pcacor$loadings[,1:20]))
test.pca$class = test$class
names(test.pca)[1:20] = paste("Comp.", 1:20, sep = "")

summary(train$class)
summary(test$class)
