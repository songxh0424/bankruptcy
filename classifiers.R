# SVM


# random forest
set.seed(1234)
fit.rf = randomForest(class ~ ., data = train)
fit.rf
pred.rf = predict(fit.rf, newdata = test[, -64])
table(test$class, pred.rf)

# boosted classification trees
set.seed(1234)
fit.bst = boosting(class ~ ., train)
pred.bst = predict(fit.bst, newdata = test[, -64])
table(test$class, pred.bst$class)

# with duplicate samples, oversampling
temp = filter(train, class == 1)
train_d = bind_rows(train, temp[rep(row.names(temp), 12), ])
set.seed(1234)
fit_d.rf = randomForest(class ~ ., data = train_d)
fit_d.rf
pred_d.rf = predict(fit_d.rf, newdata = test[, -64])
table(test$class, pred_d.rf)

set.seed(1234)
fit_d.bst = boosting(class ~ ., train_d)
pred_d.bst = predict(fit_d.bst, newdata = test[, -64])
table(test$class, pred_d.bst$class)

# undersampling


# with number of na as a predictor: boosting 45%
set.seed(1234)
fit_na.rf = randomForest(class ~ ., data = train_na)
fit_na.rf
pred_na.rf = predict(fit_na.rf, newdata = test_na[, -64])
table(test_na$class, pred_na.rf)

set.seed(1234)
fit_na.bst = boosting(class ~ ., train_na)
pred_na.bst = predict(fit_na.bst, newdata = test_na[, -64])
table(test_na$class, pred_na.bst$class)

# standardized
fit.std.rf = randomForest(class ~ ., data = train.std)
fit.std.rf
pred.std.rf = predict (fit.std.rf, newdata = test.std[, -64])
table(test.std$class, pred.std.rf)

set.seed(1234)
fit.std.bst = boosting(class ~ ., train.std)
pred.std.bst = predict(fit.std.bst, newdata = test.std[, -64])
table(test.std$class, pred.std.bst$class)
