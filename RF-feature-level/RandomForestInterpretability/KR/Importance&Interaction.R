library(randomForest)
require(randomForest)
library(randomForestExplainer)
require(randomForestExplainer)
library(pROC)
require(pROC)

setwd('G:\\华统\\MSI和MSS\\随机森林解释器的结果\\KR')
load('最小深度.rda')
load('训练集.rda')
load('R语言随机森林KR.rda')
load('交互效应.rda')




# min_depth_frame.KR <- min_depth_distribution(rf.res)
# save(min_depth_frame.KR, file = 'min_depth_frame_KR.rda')
load('min_depth_frame_KR.rda')

# importance_frame.KR <- measure_importance(rf.res)
# save(importance_frame.KR, file = 'importance_frame_KR.rda')
load('importance_frame_KR.rda')

plot_multi_way_importance(importance_frame.KR , size_measure = 'no_of_nodes', main = 'times in root and mean min depth of vars in KR')
## 鍝簺璐存爣绛?
important_variables(importance_frame.KR)
## 閲嶆柊鎸囧畾鍧愭爣杞?
plot_multi_way_importance(importance_frame.KR  ,
                          x_measure = 'accuracy_decrease',
                          y_measure = 'gini_decrease',
                          size_measure = 'p_value',
                          no_of_labels = 10,  main = 'gini decrease and accuracy decrease of vars in KR')

plot_importance_ggpairs(importance_frame.KR,
                        measures = c('mean_min_depth','gini_decrease','accuracy_decrease','times_a_root','no_of_nodes'))

(vars <- important_variables(importance_frame.KR,
                             k= 8,
                             measures = c('mean_min_depth',
                                          'no_of_nodes','times_a_root')))[c(-3,-7)]
interactions_frame <- min_depth_interactions(rf.res.KR, vars)

load('interactions_framekr.rda')


plot_min_depth_interactions(interactions_frame, main='Mean minimal depth for 30 most frequent interactions of KR')
head(interactions_frame[order(interactions_frame$occurrences, decreasing = TRUE), ])

interactions_frame.order <- interactions_frame[order(interactions_frame$occurrences, decreasing = TRUE), ]


plot_min_depth_distribution(min_depth_frame.KR, k =15,  mean_sample = 'top_trees', main = 'Distribution of minimal depth and its mean in KR')
plot_min_depth_distribution(min_depth_frame.KR, k = 15, mean_sample = 'relevant_trees', main = 'Distribution of minimal depth and its mean in KR')








nameMap <- read.csv('G:\\华统\\MSI和MSS\\随机森林解释器的结果\\STAD\\名字对照表.csv',stringsAsFactors = F)
name <- vector(mode = 'character', length=nrow(min_depth_frame.KR))
for(i in 1:nrow(min_depth_frame.KR)){
  if(min_depth_frame.KR$variable[i] == 's_var'){
    name[i] = 'S Var'
  }
  else if( min_depth_frame.KR$variable[i] %in% nameMap[,2]){
    name[i] =  nameMap[which(nameMap[,2] == min_depth_frame.KR$variable[i]), 3]
  }
  else{
    name[i] =  min_depth_frame.KR$variable[i]
  }
}
min_depth_frame.KR$variable <- name


plot_min_depth_distribution(min_depth_frame.KR, k = 15, mean_sample = 'top_trees', main = 'Distribution of minimal depth and its mean in KR')
plot_min_depth_distribution(min_depth_frame.KR, k = 15, mean_sample = 'relevant_trees', main = 'Distribution of minimal depth and its mean in KR')




# 交互作用
load('交互效应.rda')

interaction.frame.KR$root_variable <- as.character(interaction.frame.KR$root_variable)

for(i in 1:nrow(interaction.frame.KR)){
  if(interaction.frame.KR$root_variable[i] == 's_var'){
    d1 = 'S Var'
  }
  else if(interaction.frame.KR$root_variable[i] %in% nameMap[,2]){
    d1 =  nameMap[which(nameMap[,2] == interaction.frame.KR$root_variable[i]), 3]
  }
  else{
    d1 =  interaction.frame.KR$root_variable[i]
  }
  if(interaction.frame.KR$variable[i] == 's_var'){
    d2 = 'S Var'
  }
  else if(interaction.frame.KR$variable[i] %in% nameMap[,2]){
    d2 =  nameMap[which(nameMap[,2] == interaction.frame.KR$variable[i]), 3]
  }
  else{
    d2 =  interaction.frame.KR$variable[i]
  }
  
  interaction.frame.KR$variable[i] <- d2
  interaction.frame.KR$root_variable[i] <- d1
  interaction.frame.KR$interaction[i] <- paste(d1,d2, sep=':')
}


plot_min_depth_interactions(interaction.frame.KR, main = 'Mean minimal depth for 30 most frequent interactions of KR')




train.data.KR <- train.data.new
rf.res.KR <- rf.res
interactions_frame.order <- interaction.frame.KR[order(interaction.frame.KR$occurrences, decreasing = TRUE), ]



plot_predict_interaction(rf.res.KR, train.data.KR , "Nucleus..Perimeter", "immune_num")+ geom_hline(yintercept = 200)+ geom_vline(xintercept = 80) +labs(x = 'NucleusPerimeter', y = 'ImmuneCellCount',title = 'KR')

