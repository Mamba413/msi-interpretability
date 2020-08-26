setwd('G:\\��ͳ\\MSI��MSS\\���ɭ�ֽ������Ľ��\\STAD')
load('��С���.rda')

nameMap <- read.csv('G:\\��ͳ\\MSI��MSS\\���ɭ�ֽ������Ľ��\\STAD\\���ֶ��ձ�.csv',stringsAsFactors = F)
name <- vector(mode = 'character', length=nrow(min_depth_frame.STAD))
for(i in 1:nrow(min_depth_frame.STAD)){
  if(min_depth_frame.STAD$variable[i] == 's_var'){
    name[i] = 'S Var'
  }
  else if( min_depth_frame.STAD$variable[i] %in% nameMap[,2]){
    name[i] =  nameMap[which(nameMap[,2] == min_depth_frame.STAD$variable[i]), 3]
  }
  else{
    name[i] =  min_depth_frame.STAD$variable[i]
  }
}
min_depth_frame.STAD$variable <- name


plot_min_depth_distribution(min_depth_frame.STAD, k = 15, mean_sample = 'top_trees', main = 'Distribution of minimal depth and its mean in STAD')
plot_min_depth_distribution(min_depth_frame.STAD, k = 15, mean_sample = 'relevant_trees', main = 'Distribution of minimal depth and its mean in STAD')




# ��������
load('����ЧӦ.rda')

interaction.frame.STAD$root_variable <- as.character(interaction.frame.STAD$root_variable)

for(i in 1:nrow(interaction.frame.STAD)){
  if(interaction.frame.STAD$root_variable[i] == 's_var'){
    d1 = 'S Var'
  }
  else if(interaction.frame.STAD$root_variable[i] %in% nameMap[,2]){
    d1 =  nameMap[which(nameMap[,2] == interaction.frame.STAD$root_variable[i]), 3]
  }
  else{
    d1 =  interaction.frame.STAD$root_variable[i]
  }
  if(interaction.frame.STAD$variable[i] == 's_var'){
    d2 = 'S Var'
  }
  else if(interaction.frame.STAD$variable[i] %in% nameMap[,2]){
    d2 =  nameMap[which(nameMap[,2] == interaction.frame.STAD$variable[i]), 3]
  }
  else{
    d2 =  interaction.frame.STAD$variable[i]
  }
  
  interaction.frame.STAD$variable[i] <- d2
  interaction.frame.STAD$root_variable[i] <- d1
  interaction.frame.STAD$interaction[i] <- paste(d1,d2, sep=':')
}


plot_min_depth_interactions(interaction.frame.STAD, main = 'Mean minimal depth for 30 most frequent interactions of STAD')





library(randomForest)
require(randomForest)
library(randomForestExplainer)
require(randomForestExplainer)
library(pROC)
require(pROC)
library(ggplot2)
setwd('G:\\��ͳ\\MSI��MSS\\���ɭ�ֽ������Ľ��\\STAD')
load('ѵ����.rda')
load('R�������ɭ��STAD.rda')
load('����ЧӦ.rda')
train.data.STAD <- train.data.new
rf.res.STAD <- rf.res

interactions_frame.order <- interaction.frame.STAD[order(interaction.frame.STAD$occurrences, decreasing = TRUE), ]

plot_predict_interaction(rf.res.STAD, train.data.STAD , "Nucleus..Hematoxylin.OD.range", "ROI..1.00.px.per.pixel..Eosin..Haralick.Correlation..F2.")+geom_hline(yintercept = 0.5) + geom_vline(xintercept = 0.5)+ 
  labs(x='NucleusHematoxylinOD range',y='Eosin Haralick Correlation', title='')+
  theme(axis.title.x = element_text(size=20), axis.title.y = element_text(size=20), 
        axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 15) ,legend.text = element_text(size = 15), legend.title = element_text(size = 15) )


