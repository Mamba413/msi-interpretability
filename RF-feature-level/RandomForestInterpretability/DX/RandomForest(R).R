library(randomForest)
require(randomForest)
library(randomForestExplainer)
require(randomForestExplainer)
library(pROC)
require(pROC)

# type your dataset and path
dataset = 'DX'
position= 'G:\\华统\\MSI和MSS\\数据汇总8(基于7进行训练集测试集划分)\\DX'



# pipeline
left.thre <- ifelse(dataset=='DX', 16,  3)
min.split <- ifelse(dataset=='DX', 17,  2)

# read files
## 1.collor feature
setwd(position)
train.msi.hsvrgb <- read.csv('Train_MSI_RGBHSV.csv')
train.mss.hsvrgb <- read.csv('Train_MSS_RGBHSV.csv')
test.msi.hsvrgb <- read.csv('Test_MSI_RGBHSV.csv')
test.mss.hsvrgb <- read.csv('Test_MSS_RGBHSV.csv')
## 2.immune number
train.msi.immune.number <- read.csv('Train_MSI_immune.csv')
train.mss.immune.number <- read.csv('Train_MSS_immune.csv')
test.msi.immune.number <- read.csv('Test_MSI_immune.csv')
test.mss.immune.number <- read.csv('Test_MSS_immune.csv')
## 3.differentiation feature
train.msi.dif <- read.csv('Train_MSI_DIF.csv')
train.mss.dif <- read.csv('Train_MSS_DIF.csv')
test.msi.dif <- read.csv('Test_MSI_DIF.csv')
test.mss.dif <- read.csv('Test_MSS_DIF.csv')
### turn in to 0-1
train.msi.dif$spot_num <- ifelse(train.msi.dif$spot_num > 0, 1, 0)
train.mss.dif$spot_num <- ifelse(train.mss.dif$spot_num > 0, 1, 0)
test.msi.dif$spot_num <- ifelse(test.msi.dif$spot_num > 0, 1, 0)
test.msi.dif$spot_num <- ifelse(test.msi.dif$spot_num > 0, 1, 0)
## 4.GMM feature
train.msi.GMM <- read.csv('Train_MSI_GMM.csv')
train.mss.GMM <- read.csv('Train_MSS_GMM.csv')
test.msi.GMM <- read.csv('Test_MSI_GMM.csv')
test.mss.GMM <- read.csv('Test_MSS_GMM.csv')

## 5.Texture
train.msi.Texture <- read.csv('Train_MSI_Texture.csv')
train.mss.Texture <- read.csv('Train_MSS_Texture.csv')
test.msi.Texture <- read.csv('Test_MSI_Texture.csv')
test.mss.Texture <- read.csv('Test_MSS_Texture.csv')



# merge
train.MSI.whole <- cbind(train.msi.hsvrgb[2:49], train.msi.immune.number["immune_num"], train.msi.dif[2:3], 
                         subset(train.msi.GMM, select = -c(X, patient_ID)),
                         subset(train.msi.Texture, select = -c(X, patient_ID)))
train.MSS.whole <- cbind(train.mss.hsvrgb[2:49], train.mss.immune.number["immune_num"], train.mss.dif[2:3], 
                         subset(train.mss.GMM, select = -c(X, patient_ID)),
                         subset(train.mss.Texture, select = -c(X, patient_ID)))
test.MSI.whole <- cbind(test.msi.hsvrgb[2:49], test.msi.immune.number["immune_num"], test.msi.dif[2:3], 
                        subset(test.msi.GMM, select = -c(X, patient_ID)),
                        subset(test.msi.Texture, select = -c(X, patient_ID)))
test.MSS.whole <- cbind(test.mss.hsvrgb[2:49], test.mss.immune.number["immune_num"], test.mss.dif[2:3], 
                        subset(test.mss.GMM, select = -c(X, patient_ID)),
                        subset(test.mss.Texture, select = -c(X, patient_ID)))

# apply labels to them
train.MSI.Whole <- cbind(train.MSI.whole, label = rep('msi', 49984))
train.MSS.Whole <- cbind(train.MSS.whole, label = rep('mss', 78986))
test.MSI.Whole <- cbind(test.MSI.whole, label = rep('msi', 25055))
test.MSS.Whole <- cbind(test.MSS.whole, label = rep('mss', 28373))
train.data <- rbind(train.MSI.Whole, train.MSS.Whole)
test.data <- rbind(test.MSI.Whole, test.MSS.Whole)

# data clean function
data.clean <- function(whole.data){
  loc.del <- which(whole.data[,'g_var']==0|
                     whole.data[,'r_var']==0|
                     whole.data[,'b_var']==0|
                     whole.data[,'h_var']==0|
                     whole.data[,'s_var']==0|
                     whole.data[,'v_var']==0|
                     whole.data[,'immune_num']<=left.thre|
                     whole.data[,'immune_num']>=485)
  whole.data2 <- whole.data[-loc.del, ]
  whole.data3 <- subset(whole.data2, select = -c(r_var, g_var, h_var, 
                                                 b_var, r_var, g_var))
  # normalization
  var_vec1 = c('r','g','b','h','s','v')
  var_vec2 = c('mean', '25', 'median', '75')
  for(var1 in var_vec1){
    for(var2 in var_vec2){
      var_comp = paste(var1, var2, sep = '_')
      var_de = paste(var1,'var', sep = '_')
      whole.data3[var_comp] <- whole.data2[var_comp]/whole.data2[var_de]
    }
  }
  
  one.div <- whole.data3['g_kur'] / whole.data3['h_75']
  two.div <- whole.data3['immune_num'] / whole.data3['h_skew']
  
  whole.data4 <- cbind(whole.data3, one.div, two.div)
  
  whole.data5 <- subset(whole.data4, select = -c(r_range, g_range, b_range, h_range, s_range, v_range))
  return(whole.data5)
}

train.data <- data.clean(train.data)
test.data <- data.clean(test.data)

# delete patient_ID
train.data.new <- subset(train.data, select = -c(patient_ID))
test.data.new <- subset(test.data, select = -c(patient_ID))


test.name.set <- unique(test.data$patient_ID)


# random forest
rf.res <- randomForest(label ~ ., data = train.data.new, 
                       localImp = TRUE, node.size = min.split)

rf.pred <- predict(object = rf.res, newdata = test.data.new, type = 'response')




# auc of patient
score.docu <- rep(0,length(test.name.set))
real.label <- rep(0,length(test.name.set))

i=1
for (name in test.name.set){
  pos.name<-which(test.data[,'patient_ID']==name)
  
  real.label[i]<- as.numeric(test.data[pos.name[1],'label'])-1
  
  name.prediction<-rf.pred[pos.name]
  
  score<-sum(name.prediction=='mss')/length(name.prediction)
  score.docu[i]<-score
  i<-i+1
}


auc(real.label,score.docu)



# save random forest

save.path = 'G:\\华统\\MSI和MSS\\结果汇总7(跟随数据汇总)\\DX'
setwd(save.path)
save(rf.res, file = 'R语言随机森林.rda')
load('DXrf.rda')





