# codes from joyofdata
calculate_roc <- function(df, cost_of_fp, cost_of_fn, n=100) {
  tpr <- function(df, threshold) {
    sum(df$pred >= threshold & df$class == "Yes") / sum(df$class == "Yes")
  }

  fpr <- function(df, threshold) {
    sum(df$pred >= threshold & df$class == "No") / sum(df$class == "No")
  }

  cost <- function(df, threshold, cost_of_fp, cost_of_fn) {
    sum(df$pred >= threshold & df$class == "No") * cost_of_fp +
      sum(df$pred < threshold & df$class == "Yes") * cost_of_fn
  }

  roc <- data.frame(threshold = seq(0,1,length.out=n), tpr=NA, fpr=NA)
  roc$tpr <- sapply(roc$threshold, function(th) tpr(df, th))
  roc$fpr <- sapply(roc$threshold, function(th) fpr(df, th))
  roc$cost <- sapply(roc$threshold, function(th) cost(df, th, cost_of_fp, cost_of_fn))

  return(roc)
}


plot_roc <- function(roc, threshold, cost_of_fp, cost_of_fn) {
  norm_vec <- function(v) (v - min(v))/diff(range(v))

  idx_threshold = which.min(abs(roc$threshold-threshold))

  col_ramp <- colorRampPalette(c("green","orange","red","black"))(100)
  col_by_cost <- col_ramp[ceiling(norm_vec(roc$cost)*99)+1]
  p_roc <- ggplot(roc, aes(fpr,tpr)) +
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=4, alpha=0.5) +
    coord_fixed() +
    geom_line(aes(threshold,threshold), color=rgb(0,0,1,alpha=0.5)) +
    labs(title = sprintf("ROC")) + xlab("FPR") + ylab("TPR") +
    geom_hline(yintercept=roc[idx_threshold,"tpr"], alpha=0.5, linetype="dashed") +
    geom_vline(xintercept=roc[idx_threshold,"fpr"], alpha=0.5, linetype="dashed")

  p_cost <- ggplot(roc, aes(threshold, cost)) +
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=4, alpha=0.5) +
    labs(title = sprintf("cost function")) +
    geom_vline(xintercept=threshold, alpha=0.5, linetype="dashed")

  return(list(p_roc, p_cost))
}

# visualize distribution of predictions
plot_pred_type_distribution <- function(df, threshold) {
  v <- rep(NA, nrow(df))
  v <- ifelse(df$pred >= threshold & df$class == "Yes", "TP", v)
  v <- ifelse(df$pred >= threshold & df$class == "No", "FP", v)
  v <- ifelse(df$pred < threshold & df$class == "Yes", "FN", v)
  v <- ifelse(df$pred < threshold & df$class == "No", "TN", v)

  df$pred_type <- v

  ggplot(data=df, aes(x=class, y=pred)) +
    geom_violin(fill=rgb(1,1,1,alpha=0.6), color=NA) +
    geom_jitter(aes(color=pred_type), alpha=0.6) +
    geom_hline(yintercept=threshold, color="red", alpha=0.6) +
    scale_color_discrete(name = "type")
}

# cv threshold
# cv_thres_ada = function(model, thres, train = train.na, test = test.na, fold = 10, rep = 5, seed = 1234) {
#   set.seed(seed)
#   temp = lapply(1:rep, function(i) {
#     folds = cv_partition(train$class, num_folds = fold)
#     trainprob = predict(model, newdata = train[, -64], type = "prob")[, 1]
#     trainpred = ifelse(trainprob > thres, "Yes", "No")
#     testprob = predict(model, newdata = test[, -64], type = "prob")[, 1]
#     testpred = ifelse(testprob > thres, "Yes", "No")
#     traintable = table(train$class, trainpred)
#     testtable = table(test$class, testpred)
#     train.sens = traintable[1, 1] / sum(traintable[1, ])
#     train.spec = traintable[2, 2] / sum(traintable[2, ])
#     test.sens = testtable[1, 1] / sum(testtable[1, ])
#     test.spec = testtable[2, 2] / sum(testtable[2, ])
# 
#     cv_temp = lapply(folds, function(f) {
#       cv_model = train_wrap(train[f$training, ], test[f$test, ], "ada", iter = 250, maxdepth = 5, nu = 0.1)
#       prob = predict(cv_model, newdata = )
#     })
#   })
# }

cv_thres_ada = function(thres) {
  temp = filter(model.na.ada.smote$model$pred, nu == 0.1 & maxdepth == 5 & iter == 250) %>%
    mutate(pred = factor(ifelse(Yes > thres, "Yes", "No"), levels = c("Yes", "No"))) %>%
    group_by(Resample) %>%
    summarise(sens = sum(pred == "Yes" & obs == "Yes") / sum(obs == "Yes"), spec = sum(pred == "No" & obs == "No") / sum(obs == "No"))
  return(data.frame(threshold = thres, cv.sens = mean(temp$sens), cv.spec = mean(temp$spec),
         test.sens = mean(model.na.ada.smote$probs[test.na$class == "Yes"] > thres), test.spec = mean(model.na.ada.smote$probs[test.na$class == "No"] < thres)))
}

cv_thres_gbm = function(thres) {
  temp = filter(model.na.gbm.smote$model$pred, n.trees == 300 & interaction.depth == 7 & shrinkage == 0.025 & n.minobsinnode == 10) %>%
    mutate(pred = factor(ifelse(Yes > thres, "Yes", "No"), levels = c("Yes", "No"))) %>%
    group_by(Resample) %>%
    summarise(sens = sum(pred == "Yes" & obs == "Yes") / sum(obs == "Yes"), spec = sum(pred == "No" & obs == "No") / sum(obs == "No"))
  return(data.frame(threshold = thres, cv.sens = mean(temp$sens), cv.spec = mean(temp$spec),
         test.sens = mean(model.na.gbm.smote$probs[test.na$class == "Yes"] > thres), test.spec = mean(model.na.gbm.smote$probs[test.na$class == "No"] < thres)))
}

datL.ada = bind_rows(lapply(seq(0.05, 0.95, by = 0.05), cv_thres_ada)) %>%
             gather(key = legend, value = value, -threshold) %>%
             mutate(method = "AdaBoost")
datL.gbm = bind_rows(lapply(seq(0.05, 0.95, by = 0.05), cv_thres_gbm)) %>%
             gather(key = legend, value = value, -threshold) %>%
             mutate(method = "GBM")
datL = bind_rows(datL.ada, datL.gbm)



sum(model.na.ada.smote$probs[test.na$class == "No"] > 0.1)
