library(dplyr)
tr.pca = princomp(train[, -64], cor = TRUE)
varpercent = cumsum(tr.pca$sdev^2) / sum(tr.pca$sdev^2)
ggplot(data.frame(), aes(1:20, varpercent[1:20])) + geom_line()
scores = data.frame(tr.pca$scores[, 1:2])
scores$class = train$class
ggplot(scores, aes(Comp.1, Comp.2, color = class)) + geom_point(alpha = 0.5) +
  xlim(c(-1, 1)) + ylim(c(-2, 2)) # almost 99% of data points are in this region


# prediction plot
predplot = function(model, test, threshold) {
  df = data.frame(class = test$class, pred = model$probs)
  plot_pred_type_distribution(df, threshold)
}

# MDS plot of variables
dist = 1 - abs(cor(train[, -64]))
attr.mds = cmdscale(dist, k = 2)
attr.mds = as.data.frame(attr.mds)
attr.mds$label = str_sub(names(bank)[-64], start = 5)
names(attr.mds)[1:2] = c("D1", "D2")
ggplot(attr.mds, aes(D1, D2)) + geom_text(label = attr.mds$label, position = position_jitter(height = 0.03, width = 0.03))

# box plot of NA
ggplot(train.na, aes(x = class, y = numNA, color = class)) + geom_boxplot() + ylim(c(0, 5))
ggplot(train.na, aes(numNA, fill = class)) + geom_histogram()
prop1 = mean(train.na$numNA[train$class == "Yes"] > 0)
prop2 = mean(train.na$numNA[train$class == "No"] > 0)
dat.temp = data.frame(legend = c("Yes", "Yes", "No", "No"), number_of_NA = c("None", "At least 1", "None", "At least 1"),
                      proportion = c(1 - prop1, prop1, 1 - prop2, prop2))
ggplot(dat.temp, aes(x = number_of_NA, y = proportion, fill = legend)) + geom_col(position = "dodge", width = 0.5) +
  ylim(c(0, 1))

# parcoord
ggparcoord(mutate(table.smote, methods = rownames(table.orig)), scale = "globalminmax", columns = 1:6, groupColumn = 7)
ggparcoord(mutate(roc.smote, methods = rownames(table.orig)), scale = "globalminmax", columns = 1:3, groupColumn = 7)

# threshold
ggplot(datL, aes(threshold, value, color = legend, shape = legend)) + geom_point() + geom_line() + facet_grid(. ~ method)
