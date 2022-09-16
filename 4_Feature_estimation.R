
load("50_replications.RData")
library(ggplot2)
#1. Convergence plot: for showing consistency
order_score_of_chain$seed <- as.factor(order_score_of_chain$seed)
ggplot(order_score_of_chain,aes(x=iteration,y=score,color=seed))+geom_line()+
  theme_light()+xlab("Iteration")+ylab("BDeu score")


#2. MAP score
boxplot(map_score,ylab="BDeu",xlab="MAP models")


#3. The best structure

map_network <- empty.graph(rownames(output_max_graph[[which.max(map_score)]]))
amat(map_network) <- output_max_graph[[which.max(map_score)]]

g_map <- graphviz.plot(map_network,shape="rectangle")
g_map
# graph::nodeRenderInfo(g_map) <- list(fill="lightgreen", fontsize=25)
# Rgraphviz::renderGraph(g_map)

B <- 100
n <- nrow(output_graph_chain[[r]][[1]])
#Edge features estimation
library(reshape2)
library(data.table)

library(bnlearn)


for(r in 1:50){ #r <- 1
  edge_feature  <- Reduce("+",output_graph_chain[[r]][(B+1):1000])/(1000-B)
  if(r==1){
    edge_feature_df <-reshape2::melt(edge_feature)
    edge_feature_df$seed <- r
  }else{
    temp_df <- reshape2::melt(edge_feature)
    temp_df$seed <- r
    edge_feature_df <- rbind(edge_feature_df,temp_df)
  }
}

names(edge_feature_df) <- c("Xi","Xj","prob","seed")
ggplot(edge_feature_df[edge_feature_df$seed==1,], aes(x=Xj,y=Xi, fill= prob)) + 
  geom_tile()+ theme(axis.text.x = 
                       element_text(angle = 90, vjust = 0.5, 
                                    hjust=1))

# install.packages("dplyr")                       # Install dplyr package
library("dplyr") 

edge_feature_avg <- as.data.frame(edge_feature_df %>%                                        # Specify data frame
  group_by(Xi,Xj) %>%                         # Specify group indicator
  summarise_at(vars(prob),              # Specify column
               list(prob = mean)))

# edge_feature_avg$id <- 1:nrow(edge_feature_avg) #29*29


# edge_feature_2.5 <- as.data.frame(edge_feature_df %>%                                        # Specify data frame
#                                     group_by(Xi,Xj) %>%                         # Specify group indicator
#                                     summarise_at(vars(prob),              # Specify column
#                                                  list(prob = function(x) quantile(x,0.025))))
# 
# edge_feature_97.5 <- as.data.frame(edge_feature_df %>%                                        # Specify data frame
#                                     group_by(Xi,Xj) %>%                         # Specify group indicator
#                                     summarise_at(vars(prob),              # Specify column
#                                                  list(prob = function(x) quantile(x,0.975))))


# edge_feature_avg_with_ci <- merge(edge_feature_avg,edge_feature_97.5,by=c("Xi","Xj"))
# edge_feature_avg_with_ci <- merge(edge_feature_avg_with_ci,edge_feature_2.5,by=c("Xi","Xj"))
# 
# names(edge_feature_avg_with_ci) <- c("Xi","Xj","mean","U","L")
# 
# edge_feature_avg_with_ci_melt <- reshape2::melt(edge_feature_avg_with_ci,id=c("Xi","Xj"))
# 
# nrow(edge_feature_avg_with_ci_melt)
# 
# edge_feature_avg_with_ci_melt$id <- rep(1:841,3)

#heatmap

ggplot(edge_feature_avg, aes(x=Xj,y=Xi, fill= prob)) + 
  geom_tile()+ theme(axis.text.x = 
                       element_text(angle = 90, vjust = 0.5, 
                                    hjust=1))




#Bootstrap standard error
 #29*29
edge_feature_avg <- edge_feature_avg[order(edge_feature_avg$prob),]
edge_feature_avg$arc <- 1:nrow(edge_feature_avg)


ggplot(edge_feature_avg[edge_feature_avg$prob>0,],aes(x=arc,y=prob))+geom_line()+theme_light()

#bootstrap
edge_feature_df$arc <- rep(1:nrow(edge_feature_avg),50)
for(e in 1:nrow(edge_feature_avg)){ #e<-34
  temp_sample <- edge_feature_df[edge_feature_df$arc==e,]
  #bootstrap 100 times for standard error
  temp_prob_est <- vector()
  for(b in 1:1000){
    temp_prob_est[b] <- mean(temp_sample[sample(1:50,50,replace=TRUE),"prob"])
  }
  
  edge_feature_avg[edge_feature_avg$Xi==temp_sample$Xi[1]&
                     edge_feature_avg$Xj==temp_sample$Xj[1],"se"] <- sd(temp_prob_est)
  edge_feature_avg[edge_feature_avg$Xi==temp_sample$Xi[1]&
                     edge_feature_avg$Xj==temp_sample$Xj[1],"D"] <- quantile(temp_prob_est,0.025)
  edge_feature_avg[edge_feature_avg$Xi==temp_sample$Xi[1]&
                     edge_feature_avg$Xj==temp_sample$Xj[1],"U"] <- quantile(temp_prob_est,0.975)
}

ggplot(edge_feature_avg[edge_feature_avg$prob>0,],aes(x=arc,y=prob))+geom_line()+theme_light()+
  geom_line(aes(x=arc,y=U),linetype="dashed")+
  geom_line(aes(x=arc,y=D),linetype="dashed")+xlab("Arcs")+ylab("Posterior probability")




#Summary of the probability
hist(edge_feature_avg[edge_feature_avg$prob>0,"prob"],main="",xlab="Posterior probability") 
sum(edge_feature_avg[edge_feature_avg$prob>0,"prob"]>0.8) #13 edge features have high probability
edge_feature_avg[edge_feature_avg$prob>0.8,] #Then, we can do model averaging.

edge_feature_avg[edge_feature_avg$Xj=="Biopsy",]
edge_feature_avg[edge_feature_avg$Xi=="Biopsy",] 

#Only with probability 11.3%, the biopsy would have indication to the HPV: The test is not very nice.
#Model averaging:

avg_arc_df <- edge_feature_avg[edge_feature_avg$prob>0.1,c("Xi","Xj")]
avg_arc_df$Xi <- as.character(avg_arc_df$Xi)
avg_arc_df$Xj <- as.character(avg_arc_df$Xj)
nrow(edge_feature_avg[edge_feature_avg$prob>0.1,]) #49 arcs

#If undirected, drop one arc with lower score. If formed cycle, drop the arc with lower score also.
# avg_arc_df

avg_network <- empty.graph(rownames(output_max_graph[[which.max(map_score)]]))
arcs(avg_network) <- avg_arc_df

g_map <- graphviz.plot(avg_network,shape="rectangle")
graph::nodeRenderInfo(g_map) <- list(fill="lightgreen", fontsize=20)
Rgraphviz::renderGraph(g_map)


#map analysis
map_network <- empty.graph(rownames(output_max_graph[[which.max(map_score)]]))
amat(map_network) <- output_max_graph[[which.max(map_score)]]

g_map <- graphviz.plot(map_network,shape="rectangle")
graph::nodeRenderInfo(g_map) <- list(fill="lightgreen", fontsize=20)
Rgraphviz::renderGraph(g_map)


map_fit <- bn.fit(map_network,cancer_df)

# predict(map_fit,"Dx.HPV",data.frame(Biopsy=1))

cpquery(map_fit,(Dx.HPV==1),(Biopsy==1),n=10^6)
cpquery(map_fit,(Dx.HPV==0),(Biopsy==0),n=10^6)



#Q: Smoking and HPV?
edge_feature_avg[edge_feature_avg$Xi=="Smokes" & edge_feature_avg$Xj=="Biopsy",] 
edge_feature_avg[edge_feature_avg$Xi=="Smokes" & edge_feature_avg$Xj=="Dx.HPV",] 

#Q: Smoking and cancer?
edge_feature_avg[edge_feature_avg$Xi=="Smokes" & edge_feature_avg$Xj=="Dx",] #Very high implication
#map probability
map_fit$Dx
cpquery(map_fit,(Dx==0),(Smokes==0),n=10^6)
cpquery(map_fit,(Dx==0),(Smokes==1),n=10^6)


cpquery(map_fit,(Dx.HPV==1),(Smokes==0),n=10^6)
cpquery(map_fit,(Dx.HPV==1),(Smokes==1),n=10^6) #Not very meaningful here..

cpquery(map_fit,(Dx.Cancer==1),(Smokes==0),n=10^6)
cpquery(map_fit,(Dx.Cancer==1),(Smokes==1),n=10^6) #Not very meaningful here..


cpquery(map_fit,(Dx.Cancer==1),(IUD==0),n=10^6) #Without using IUD, the probability of having cancer is higher.
cpquery(map_fit,(Dx.Cancer==1),(IUD==1),n=10^6)

cpquery(map_fit,(Biopsy==1),(Dx.HPV==0),n=10^6) #Without using IUD, the probability of having cancer is higher.
cpquery(map_fit,(Biopsy==1),(Dx.HPV==1),n=10^6)

#CPDAG
map_network <- empty.graph(rownames(output_max_graph[[which.max(map_score)]]))
amat(map_network) <- output_max_graph[[which.max(map_score)]]

g_map <- graphviz.plot(cpdag(map_network),shape="rectangle")
graph::nodeRenderInfo(g_map) <- list(fill="lightgreen", fontsize=20)
Rgraphviz::renderGraph(g_map)


#heatmap again

ggplot(edge_feature_avg, aes(x=Xj,y=Xi, fill= prob)) + 
  geom_tile()+ theme(axis.text.x = 
                       element_text(angle = 90, vjust = 0.5, 
                                    hjust=1))

edge_feature_avg[edge_feature_avg$Xi=="Dx.HPV",]
edge_feature_avg[edge_feature_avg$Xi=="Smokes",]
edge_feature_avg[edge_feature_avg$Xi=="IUD",]


edge_feature_avg[edge_feature_avg$Xi=="IUD",]

edge_feature_avg[edge_feature_avg$Xi=="Dx.HPV",]

cpquery(map_fit,(Dx.Cancer==1),(Dx.HPV==0),n=10^6) #Without using IUD, the probability of having cancer is higher.
cpquery(map_fit,(Dx.Cancer==1),(Dx.HPV==1),n=10^6)


cpquery(map_fit,(Dx.Cancer==1),(Dx.HPV==0),n=10^6) #Without using IUD, the probability of having cancer is higher.
cpquery(map_fit,(Dx.Cancer==1),(Dx.HPV==1),n=10^6)

cpquery(map_fit,(Dx.HPV==1),(Smokes==0),n=10^6) #Without using IUD, the probability of having cancer is higher.
cpquery(map_fit,(Dx.HPV==1),(Smokes==1),n=10^6)

a0 <- 0.02107726
a1 <- 0.02112072

a1/a0

a1/(1-a1)/(a0/(1-a0))

edge_feature_avg[edge_feature_avg$Xi=="IUD",]

