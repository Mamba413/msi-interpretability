# ����DX ROC����
path1 <- 'G:\\��ͳ\\MSI��MSS\\�������7(�������ݻ���)\\DX\\R�������ɭ��.rda'
load(path1)




obj <- roc(real.label, score.docu, ci=TRUE, ci.method = 'bootstrap', plot=FALSE)
obj$ci
ciobj <- ci.se(obj, specificities=seq(0, 1, l=25),conf.level = 0.6)
dat.ci <- data.frame(x = as.numeric(rownames(ciobj)),
                     lower = ciobj[, 1],
                     upper = ciobj[, 3])

ggroc(obj) + theme_minimal() + geom_abline(slope=1, intercept = 1, linetype = "dashed", alpha=0.7, color = "grey") + coord_equal() + 
  geom_ribbon(data = dat.ci, aes(x = x, ymin = lower, ymax = upper), fill = "steelblue", alpha= 0.2) + ggtitle('ROC curve of DX')








# ����KR ROC����
path1 <- 'G:\\��ͳ\\MSI��MSS\\�������7(�������ݻ���)\\KR\\R�������ɭ��KR.rda'
load(path1)


# �������ݼ�������ԭ������
# �Թ����ɭ��
# ֻ��Ԥ��
# �õ�scorehe reallabel

obj <- roc(real.label, score.docu, ci=TRUE, ci.method = 'bootstrap', plot=FALSE)
obj$ci
ciobj <- ci.se(obj, specificities=seq(0, 1, l=25),conf.level = 0.6)
dat.ci <- data.frame(x = as.numeric(rownames(ciobj)),
                     lower = ciobj[, 1],
                     upper = ciobj[, 3])

ggroc(obj) + theme_minimal() + geom_abline(slope=1, intercept = 1, linetype = "dashed", alpha=0.7, color = "grey") + coord_equal() + 
  geom_ribbon(data = dat.ci, aes(x = x, ymin = lower, ymax = upper), fill = "steelblue", alpha= 0.2) + ggtitle('ROC curve of KR')











# STAD auc
path1 <- 'G:\\��ͳ\\MSI��MSS\\�������7(�������ݻ���)\\STAD\\R�������ɭ��STAD.rda'
load(path1)


# �������ݼ�������ԭ������
# �Թ����ɭ��
# ֻ��Ԥ��
# �õ�scorehe reallabel

obj <- roc(real.label, score.docu, ci=TRUE, ci.method = 'bootstrap', plot=FALSE)
obj$ci
ciobj <- ci.se(obj, specificities=seq(0, 1, l=25),conf.level = 0.6)
dat.ci <- data.frame(x = as.numeric(rownames(ciobj)),
                     lower = ciobj[, 1],
                     upper = ciobj[, 3])

ggroc(obj) + theme_minimal() + geom_abline(slope=1, intercept = 1, linetype = "dashed", alpha=0.7, color = "grey") + coord_equal() + 
  geom_ribbon(data = dat.ci, aes(x = x, ymin = lower, ymax = upper), fill = "steelblue", alpha= 0.2) + ggtitle('ROC curve of STAD')


