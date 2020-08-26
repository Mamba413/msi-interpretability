setwd('G:\\华统\\MSI和MSS\\随机森林解释器的结果\\DX')
library(randomForest)
require(randomForest)
library(randomForestExplainer)
require(randomForestExplainer)
library(pROC)
require(pROC)
library(ggplot2)
load('训练集.rda')
load('R语言随机森林.rda')
load('交互效应.rda')


#min_depth_frame.DX <- min_depth_distribution(rf.res)
#save(min_depth_frame.DX, file = 'min_depth_frame_DX.rda')
load('min_depth_frame_DX.rda')

plot_min_depth_distribution(min_depth_frame.DX, k = 15, mean_sample = 'top_trees', main = 'Distribution of minimal depth and its mean in DX')
plot_min_depth_distribution(min_depth_frame.DX, k = 15, mean_sample = 'relevant_trees', main = 'Distribution of minimal depth and its mean in DX')


# importance_frame.DX <- measure_importance(rf.res)
# save(importance_frame.DX, file = 'importance_frame_DX.rda')
load('importance_frame_DX.rda')


plot_multi_way_importance(importance_frame.DX , size_measure = 'no_of_nodes', main = 'times in root and mean min depth of vars in DX')

important_variables(importance_frame)

plot_multi_way_importance(importance_frame.DX  ,
                          x_measure = 'accuracy_decrease',
                          y_measure = 'gini_decrease',
                          size_measure = 'p_value',
                          no_of_labels = 10,  main = 'gini decrease and accuracy decrease of vars in DX')

plot_importance_ggpairs(importance_frame.DX,
                        measures = c('mean_min_depth','gini_decrease','accuracy_decrease','times_a_root','no_of_nodes'))

(vars <- important_variables(importance_frame.DX,
                             k= 4,
                             measures = c('mean_min_depth',
                                          'no_of_nodes','times_a_root')))


interactions_frame <- min_depth_interactions(rf.res, vars)

# save(interactions_frame, file = 'interactions_frameDX.rda')
load('interactions_frameDX.rda')

head(interactions_frame[order(interactions_frame$occurrences, decreasing = TRUE), ])
interactions_frame.order <- interactions_frame[order(interactions_frame$occurrences, decreasing = TRUE), ]

plot_min_depth_interactions(interactions_frame, main = 'Mean minimal depth for 30 most frequent interactions of DX')



# 读取随机森林解释器
setwd('G:\\华统\\MSI和MSS\\随机森林解释器的结果\\DX')
load('最小深度.rda')


nameMap <- read.csv('G:\\华统\\MSI和MSS\\随机森林解释器的结果\\DX\\名字对照表.csv',stringsAsFactors = F)
name <- vector(mode = 'character', length=nrow(min_depth_frame.DX))
for(i in 1:nrow(min_depth_frame.DX)){
  if(min_depth_frame.DX$variable[i] == 's_var'){
    name[i] = 'S Var'
  }
  else if( min_depth_frame.DX$variable[i] %in% nameMap[,2]){
    name[i] =  nameMap[which(nameMap[,2] == min_depth_frame.DX$variable[i]), 3]
  }
  else{
    name[i] =  min_depth_frame.DX$variable[i]
  }
}
min_depth_frame.DX$variable <- name



plot_min_depth_distribution(min_depth_frame.DX, k = 15, mean_sample = 'top_trees', main = 'Distribution of minimal depth and its mean in DX')
plot_min_depth_distribution(min_depth_frame.DX, k = 15, mean_sample = 'relevant_trees', main = 'Distribution of minimal depth and its mean in DX')





# 交互作用
load('交互效应.rda')

interaction.frame.DX$root_variable <- as.character(interaction.frame.DX$root_variable)

for(i in 1:nrow(interaction.frame.DX)){
  if(interaction.frame.DX$root_variable[i] == 's_var'){
    d1 = 'S Var'
  }
  else if(interaction.frame.DX$root_variable[i] %in% nameMap[,2]){
    d1 =  nameMap[which(nameMap[,2] == interaction.frame.DX$root_variable[i]), 3]
  }
  else{
    d1 =  interaction.frame.DX$root_variable[i]
  }
  if(interaction.frame.DX$variable[i] == 's_var'){
    d2 = 'S Var'
  }
  else if(interaction.frame.DX$variable[i] %in% nameMap[,2]){
    d2 =  nameMap[which(nameMap[,2] == interaction.frame.DX$variable[i]), 3]
  }
  else{
    d2 =  interaction.frame.DX$variable[i]
  }
  
  interaction.frame.DX$variable[i] <- d2
  interaction.frame.DX$root_variable[i] <- d1
  interaction.frame.DX$interaction[i] <- paste(d1,d2, sep=':')
}

interaction.frame.DX$variable <- name2
interaction.frame.DX$root_variable <- name1

plot_min_depth_interactions(interaction.frame.DX, main = 'Mean minimal depth for 30 most frequent interactions of DX')



train.data.DX <- train.data.new
rf.res.DX <- rf.res
# plot_predict_interaction(rf.res.DX, train.data.DX , "Cell..Hematoxylin.OD.max", "b_25")
# plot_predict_interaction(rf.res.DX, train.data.DX , "Cell..Max.caliper", "b_25")
plot_predict_interaction(rf.res.DX, train.data.DX , "count", "Cell..Max.caliper") + geom_hline(yintercept = 20)+ 
  geom_vline(xintercept = 220) +labs(x = 'Tumor Count', y = 'Cell Max Caliper',title = '')+
  theme(axis.title.x = element_text(size=20), axis.title.y = element_text(size=20), 
        axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 15) ,legend.text = element_text(size = 15), legend.title = element_text(size = 15) )




