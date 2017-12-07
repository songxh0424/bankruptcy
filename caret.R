library(doMC)
library(caret)
library(knitr)

# wrapper function
train_wrap = function(train, test, method, metric = "ROC", sampling = NULL, cv = FALSE, numcore = 30, seed = 1234, ...) {
  t1 = proc.time()
  registerDoMC(cores = numcore)
  set.seed(seed)

  ctrl = trainControl(sampling = sampling)
  if(cv) {ctrl$method = "repeatedcv"; ctrl$number = 10; ctrl$repeats = 5}
  if(metric == "ROC") {ctrl$savePredictions = TRUE; ctrl$classProbs = TRUE; ctrl$summaryFunction = twoClassSummary}

  model = train(class ~ ., data = train, method = method, trControl = ctrl, metric = metric, ...)
  pred = predict(model, newdata = test[, -64])
  probs = predict(model, newdata = test[, -64], type = "prob")[, 1]
  trainpred = predict(model, newdata = train[, -64])
  testtb = table(test$class, pred)
  traintb = table(train$class, trainpred)
  time = proc.time() - t1
  df = data.frame(class = train$class, pred = predict(model, newdata = train[, -64], type = "prob")[, 1])
  roc = calculate_roc(df, 1, sum(train$class == "No") / sum(train$class == "Yes"))
  plots = plot_roc(roc, 0.5, 1, sum(train$class == "No") / sum(train$class == "Yes"))
  plot.train = plot_pred_type_distribution(df, 0.5)
  df = data.frame(class = test$class, pred = probs)
  plot.test = plot_pred_type_distribution(df, 0.5)
  return(list(model = model, pred = pred, probs = probs, roc = roc, table = testtb,
              traintable = traintb, time = time, plot.roc = plots[[1]], plot.cost = plots[[2]],
              plot.train = plot.train, plot.test = plot.test))
}




# rf with rose: 30%
# model.rf.rose = train_wrap(train, test, "rf", sampling = "rose", cv = TRUE, tuneLength = 10)

## without any special treatment
model.lda = train_wrap(train, test, "lda", cv = TRUE, metric = "Accuracy")
model.logis = train_wrap(train, test, "glm", cv = TRUE, metric = "Accuracy", family = "binomial")
model.knn = train_wrap(train, test, "knn", cv = TRUE, metric = "Accuracy")
model.svm = train_wrap(train, test, "svmRadial", cv = TRUE, metric = "Accuracy")
model.nn = train_wrap(train, test, "nnet", cv = TRUE, metric = "Accuracy")
model.rf = train_wrap(train, test, "rf", cv = TRUE, metric = "Accuracy", tuneLength = 10)
grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
model.ada = train_wrap(train, test, "ada", cv = TRUE, metric = "Accuracy", tuneGrid = grid)
grid = expand.grid(n.trees = (2:6) * 50, interaction.depth = 3:6, shrinkage = 0.025 * (1:4), n.minobsinnode = 10)
model.gbm = train_wrap(train, test, "gbm", cv = TRUE, metric = "Accuracy", tuneGrid = grid)

## with PCA
model.lda.pca = train_wrap(train.pca, test.pca, "lda", cv = TRUE, metric = "Accuracy")
model.logis.pca = train_wrap(train.pca, test.pca, "glm", cv = TRUE, metric = "Accuracy", family = "binomial")
model.knn.pca = train_wrap(train.pca, test.pca, "knn", cv = TRUE, metric = "Accuracy")
model.svm.pca = train_wrap(train.pca, test.pca, "svmRadial", cv = TRUE, metric = "Accuracy")
model.nn.pca = train_wrap(train.pca, test.pca, "nnet", cv = TRUE, metric = "Accuracy")
model.rf.pca = train_wrap(train.pca, test.pca, "rf", cv = TRUE, metric = "Accuracy", tuneLength = 10)
grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
model.ada.pca = train_wrap(train.pca, test.pca, "ada", cv = TRUE, metric = "Accuracy", tuneGrid = grid)
grid = expand.grid(n.trees = (2:6) * 50, interaction.depth = 3:6, shrinkage = 0.025 * (1:4), n.minobsinnode = 10)
model.gbm.pca = train_wrap(train.pca, test.pca, "gbm", cv = TRUE, metric = "Accuracy", tuneGrid = grid)


## with na dataset, smote sampling
# model.na.adabag.smote = train_wrap(train.na, test.na, "AdaBag", sampling = "smote", tuneGrid = grid, cv = TRUE)
model.na.logis.smote = train_wrap(train.na, test.na, "glm", sampling = "smote", cv = TRUE, family = "binomial")
model.na.knn.smote = train_wrap(train.na, test.na, "knn", sampling = "smote", cv = TRUE, tuneLength = 10)
model.na.lda.smote = train_wrap(train.na, test.na, "lda", sampling = "smote", cv = TRUE)
model.na.svm.smote = train_wrap(train.na, test.na, "svmRadial", sampling = "smote", cv = TRUE)
model.na.nn.smote = train_wrap(train.na, test.na, "nnet", sampling = "smote", cv = TRUE)
model.na.rf.smote = train_wrap(train.na, test.na, "rf", sampling = "smote", tuneLength = 10, cv = TRUE)
grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
model.na.ada.smote = train_wrap(train.na, test.na, "ada", sampling = "smote", tuneGrid = grid, cv = TRUE)
grid = expand.grid(n.trees = (2:6) * 50, interaction.depth = 3:7, shrinkage = 0.025 * (1:4), n.minobsinnode = 10)
model.na.gbm.smote = train_wrap(train.na, test.na, "gbm", sampling = "smote", tuneGrid = grid, cv = TRUE)

## upsampling
model.na.lda.up = train_wrap(train.na, test.na, "lda", sampling = "up", cv = TRUE)
model.na.logis.up = train_wrap(train.na, test.na, "glm", sampling = "up", cv = TRUE, family = "binomial")
model.na.knn.up = train_wrap(train.na, test.na, "knn", sampling = "up", cv = TRUE, tuneLength = 10)
model.na.svm.up = train_wrap(train.na, test.na, "svmRadial", sampling = "up", cv = TRUE)
model.na.nn.up = train_wrap(train.na, test.na, "nnet", sampling = "up", cv = TRUE)
model.na.rf.up = train_wrap(train.na, test.na, "rf", sampling = "up", cv = TRUE, tuneLength = 10)
grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
model.na.ada.up = train_wrap(train.na, test.na, "ada", sampling = "up", cv = TRUE, tuneGrid = grid)
grid = expand.grid(n.trees = (2:6) * 50, interaction.depth = 3:7, shrinkage = 0.025 * (1:4), n.minobsinnode = 10)
model.na.gbm.up = train_wrap(train.na, test.na, "gbm", sampling = "up", tuneGrid = grid, cv = TRUE)

## downsampling
model.na.lda.down = train_wrap(train.na, test.na, "lda", sampling = "down", cv = TRUE)
model.na.logis.down = train_wrap(train.na, test.na, "glm", sampling = "down", cv = TRUE, family = "binomial")
model.na.knn.down = train_wrap(train.na, test.na, "knn", sampling = "down", cv = TRUE, tuneLength = 10)
model.na.svm.down = train_wrap(train.na, test.na, "svmRadial", sampling = "down", cv = TRUE)
model.na.nn.down = train_wrap(train.na, test.na, "nnet", sampling = "down", cv = TRUE)
model.na.rf.down = train_wrap(train.na, test.na, "rf", sampling = "down", cv = TRUE, tuneLength = 10)
grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
model.na.ada.down = train_wrap(train.na, test.na, "ada", sampling = "down", cv = TRUE, tuneGrid = grid)
grid = expand.grid(n.trees = (2:6) * 50, interaction.depth = 3:7, shrinkage = 0.025 * (1:4), n.minobsinnode = 10)
model.na.gbm.down = train_wrap(train.na, test.na, "gbm", sampling = "down", tuneGrid = grid, cv = TRUE)

save.image("project.RData")
q(save = "no")

## with PCA
# random forest with smote sampling and ROC: 72.4%, 91%
# model.pca.rf.smote = train_wrap(train.pca, test.pca, "rf", sampling = "smote", tuneLength = 10, cv = TRUE)

# random forest with down sampling and ROC: 84%, yet only 80% for not bankrupt
# model.pca.rf.down = train_wrap(train.pca, test.pca, "rf", sampling = "down", cv = TRUE, tuneLength = 10)

# ada: 71%, 93%, has better precision(TP / (TP + FP)), although has display error
# grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
# model.pca.ada.smote = train_wrap(train.pca, test.pca, "ada", sampling = "smote", tuneGrid = grid, cv = TRUE)

# gbm
# grid = expand.grid(n.trees = (2:5) * 50, interaction.depth = 3:6, shrinkage = 0.1, n.minobsinnode = 10)
# model.pca.gbm.smote = train_wrap(train.pca, test.pca, "gbm", sampling = "smote", tuneGrid = grid, cv = TRUE)

# adabag with smote
# grid = expand.grid(mfinal = (1:4) * 50, maxdepth = 2:5)
# model.adabag.smote = train_wrap(train.pca, test.pca, "AdaBag", sampling = "smote", tuneGrid = grid, cv = TRUE)


## with original dataset
# random forest with smote sampling and ROC: 65%, 90%
# model.rf.smote = train_wrap(train, test, "rf", sampling = "smote", tuneLength = 10, cv = TRUE)

# random forest with down sampling and ROC: 77%, yet only 81% for not bankrupt
# model.rf.down = train_wrap(train, test, "rf", sampling = "down", cv = TRUE, tuneLength = 10)

# ada: 67%, 92%, has better precision(TP / (TP + FP))
# grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
# model.ada.smote = train_wrap(train, test, "ada", sampling = "smote", tuneGrid = grid, cv = TRUE)

# gbm
# grid = expand.grid(n.trees = (2:5) * 50, interaction.depth = 3:6, shrinkage = 0.1, n.minobsinnode = 10)
# model.gbm.smote = train_wrap(train, test, "gbm", sampling = "smote", tuneGrid = grid, cv = TRUE)

# adabag with smote
# grid = expand.grid(mfinal = (1:4) * 50, maxdepth = 2:5)
# model.adabag.smote = train_wrap(train, test, "AdaBag", sampling = "smote", tuneGrid = grid, cv = TRUE)

grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
model.ada = train_wrap(train, test, "ada", cv = TRUE, metric = "Accuracy", tuneGrid = grid)
grid = expand.grid(n.trees = (2:6) * 50, interaction.depth = 3:6, shrinkage = 0.025 * (1:4), n.minobsinnode = 10)
model.gbm = train_wrap(train, test, "gbm", cv = TRUE, metric = "Accuracy", tuneGrid = grid)

grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
model.ada.pca = train_wrap(train.pca, test.pca, "ada", cv = TRUE, metric = "Accuracy", tuneGrid = grid)
grid = expand.grid(n.trees = (2:6) * 50, interaction.depth = 3:6, shrinkage = 0.025 * (1:4), n.minobsinnode = 10)
model.gbm.pca = train_wrap(train.pca, test.pca, "gbm", cv = TRUE, metric = "Accuracy", tuneGrid = grid)

grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
model.na.ada.smote = train_wrap(train.na, test.na, "ada", sampling = "smote", tuneGrid = grid, cv = TRUE)
grid = expand.grid(n.trees = (2:6) * 50, interaction.depth = 3:7, shrinkage = 0.025 * (1:4), n.minobsinnode = 10)
model.na.gbm.smote = train_wrap(train.na, test.na, "gbm", sampling = "smote", tuneGrid = grid, cv = TRUE)

model.na.lda.up = train_wrap(train.na, test.na, "lda", sampling = "up", cv = TRUE)
model.na.logis.up = train_wrap(train.na, test.na, "glm", sampling = "up", cv = TRUE, family = "binomial")
model.na.knn.up = train_wrap(train.na, test.na, "knn", sampling = "up", cv = TRUE, tuneLength = 10)
model.na.svm.up = train_wrap(train.na, test.na, "svmRadial", sampling = "up", cv = TRUE)
model.na.nn.up = train_wrap(train.na, test.na, "nnet", sampling = "up", cv = TRUE)
model.na.rf.up = train_wrap(train.na, test.na, "rf", sampling = "up", cv = TRUE, tuneLength = 10)
grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
model.na.ada.up = train_wrap(train.na, test.na, "ada", sampling = "up", cv = TRUE, tuneGrid = grid)
grid = expand.grid(n.trees = (2:6) * 50, interaction.depth = 3:7, shrinkage = 0.025 * (1:4), n.minobsinnode = 10)
model.na.gbm.up = train_wrap(train.na, test.na, "gbm", sampling = "up", tuneGrid = grid, cv = TRUE)

grid = expand.grid(iter = (3:6) * 50, maxdepth = 3:7, nu = 0.1)
model.na.ada.down = train_wrap(train.na, test.na, "ada", sampling = "down", cv = TRUE, tuneGrid = grid)
grid = expand.grid(n.trees = (2:6) * 50, interaction.depth = 3:7, shrinkage = 0.025 * (1:4), n.minobsinnode = 10)
model.na.gbm.down = train_wrap(train.na, test.na, "gbm", sampling = "down", tuneGrid = grid, cv = TRUE)

save.image("project.RData")
q(save = "no")
