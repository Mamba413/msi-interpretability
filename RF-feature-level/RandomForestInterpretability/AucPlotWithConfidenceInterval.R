# 绘制DX ROC曲线
path1 <- 'G:\\华统\\MSI和MSS\\结果汇总7(跟随数据汇总)\\DX\\R语言随机森林.rda'
load(path1)




obj <- roc(real.label, score.docu, ci=TRUE, ci.method = 'bootstrap', plot=FALSE)
obj$ci
ciobj <- ci.se(obj, specificities=seq(0, 1, l=25),conf.level = 0.6)
dat.ci <- data.frame(x = as.numeric(rownames(ciobj)),
                     lower = ciobj[, 1],
                     upper = ciobj[, 3])

ggroc(obj) + theme_minimal() + geom_abline(slope=1, intercept = 1, linetype = "dashed", alpha=0.7, color = "grey") + coord_equal() + 
  geom_ribbon(data = dat.ci, aes(x = x, ymin = lower, ymax = upper), fill = "steelblue", alpha= 0.2) + ggtitle('ROC curve of DX')








# 绘制KR ROC曲线
path1 <- 'G:\\华统\\MSI和MSS\\结果汇总7(跟随数据汇总)\\KR\\R语言随机森林KR.rda'
load(path1)


# 生成数据集都是用原来代码
# 略过随机森林
# 只用预测
# 得到scorehe reallabel

obj <- roc(real.label, score.docu, ci=TRUE, ci.method = 'bootstrap', plot=FALSE)
obj$ci
ciobj <- ci.se(obj, specificities=seq(0, 1, l=25),conf.level = 0.6)
dat.ci <- data.frame(x = as.numeric(rownames(ciobj)),
                     lower = ciobj[, 1],
                     upper = ciobj[, 3])

ggroc(obj) + theme_minimal() + geom_abline(slope=1, intercept = 1, linetype = "dashed", alpha=0.7, color = "grey") + coord_equal() + 
  geom_ribbon(data = dat.ci, aes(x = x, ymin = lower, ymax = upper), fill = "steelblue", alpha= 0.2) + ggtitle('ROC curve of KR')











# STAD auc
path1 <- 'G:\\华统\\MSI和MSS\\结果汇总7(跟随数据汇总)\\STAD\\R语言随机森林STAD.rda'
load(path1)


# 生成数据集都是用原来代码
# 略过随机森林
# 只用预测
# 得到scorehe reallabel

obj <- roc(real.label, score.docu, ci=TRUE, ci.method = 'bootstrap', plot=FALSE)
obj$ci
ciobj <- ci.se(obj, specificities=seq(0, 1, l=25),conf.level = 0.6)
dat.ci <- data.frame(x = as.numeric(rownames(ciobj)),
                     lower = ciobj[, 1],
                     upper = ciobj[, 3])

ggroc(obj) + theme_minimal() + geom_abline(slope=1, intercept = 1, linetype = "dashed", alpha=0.7, color = "grey") + coord_equal() + 
  geom_ribbon(data = dat.ci, aes(x = x, ymin = lower, ymax = upper), fill = "steelblue", alpha= 0.2) + ggtitle('ROC curve of STAD')



