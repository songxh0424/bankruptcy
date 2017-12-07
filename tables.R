perform = function(model) {
  attach(model)
  temp = data.frame("train accu" = (traintable[1, 1] + traintable[2, 2]) / sum(traintable), "train sens" = traintable[1, 1] / sum(traintable[1, ]), "train spec" = traintable[2, 2] / sum(traintable[2, ]),
                    "test accu" = (table[1, 1] + table[2, 2]) / sum(table), "test sens" = table[1, 1] / sum(table[1, ]), "test spec" = table[2, 2] / sum(table[2, ]))
  detach(model)
  return(temp)
}

cvvalues = function(model) {
  idx = which.max(model$model$results$ROC)
  n = dim(model$model$results)[2]
  return(model$model$results[idx, (n-5):n])
}

# original data
model.orig = list(model.lda, model.logis, model.knn, model.svm, model.nn, model.rf, model.ada, model.gbm)
table.orig = bind_rows(lapply(model.orig, perform))
table.orig = round(table.orig, digits = 4)
table.orig[7, ] = 1 - table.orig[7, ]
metnames = c("LDA", "Logistic", "kNN", "SVM", "NN", "RF", "AdaBoost", "GBM")
rownames(table.orig) = metnames
kable(table.orig)

# pca
model.pca = list(model.lda.pca, model.logis.pca, model.knn.pca, model.svm.pca, model.nn.pca, model.rf.pca, model.ada.pca, model.gbm.pca)
table.pca = bind_rows(lapply(model.pca, perform))
table.pca = round(table.pca, digits = 4)
table.pca[7, ] = 1 - table.pca[7, ]
rownames(table.pca) = metnames
kable(table.pca)

# smote
model.smote = list(model.na.lda.smote, model.na.logis.smote, model.na.knn.smote, model.na.svm.smote, model.na.nn.smote, model.na.rf.smote, model.na.ada.smote, model.na.gbm.smote)
table.smote = bind_rows(lapply(model.smote, perform))
table.smote = round(table.smote, digits = 4)
table.smote[7, ] = 1 - table.smote[7, ]
roc.smote = bind_rows(lapply(model.smote, cvvalues))
roc.smote = round(roc.smote, digits = 4)
roc.smote[7, 2:3] = 1 - roc.smote[7, 2:3]
rownames(table.smote) = metnames
rownames(roc.smote) = metnames
kable(table.smote)
kable(roc.smote)

# up
model.up = list(model.na.lda.up, model.na.logis.up, model.na.knn.up, model.na.svm.up, model.na.nn.up, model.na.rf.up, model.na.ada.up, model.na.gbm.up)
table.up = bind_rows(lapply(model.up, perform))
table.up = round(table.up, digits = 4)
table.up[7, ] = 1 - table.up[7, ]
roc.up = bind_rows(lapply(model.up, cvvalues))
roc.up = round(roc.up, digits = 4)
roc.up[7, 2:3] = 1 - roc.up[7, 2:3]
rownames(table.up) = metnames
rownames(roc.up) = metnames
kable(table.up)
kable(roc.up)

# down
model.down = list(model.na.lda.down, model.na.logis.down, model.na.knn.down, model.na.svm.down, model.na.nn.down, model.na.rf.down, model.na.ada.down, model.na.gbm.down)
table.down = bind_rows(lapply(model.down, perform))
table.down = round(table.down, digits = 4)
table.down[7, ] = 1 - table.down[7, ]
roc.down = bind_rows(lapply(model.down, cvvalues))
roc.down = round(roc.down, digits = 4)
roc.down[7, 2:3] = 1 - roc.down[7, 2:3]
rownames(table.down) = metnames
rownames(roc.down) = metnames
kable(table.down)
kable(roc.down)
