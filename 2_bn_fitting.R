library(imbalance)
store_df <- list()
for(seed in 1:10){
set.seed(seed)
prediction_vec <- vector()


cancer_df_dc <- read.csv("cancer_data_discretized.csv",colClasses="factor")
cancer_df_dc_copy <- cancer_df_dc

lapply(cancer_df_dc_copy,table)
#drop something
cancer_df_dc_copy$`STDs..number.`[cancer_df_dc_copy$`STDs..number.`==4] <- 3
cancer_df_dc_copy$`STDs..number.` <- as.factor(as.character(cancer_df_dc_copy$`STDs..number.`))

cancer_df_dc_copy$STDs.pelvic.inflammatory.disease <- NULL
cancer_df_dc_copy$STDs.genital.herpes <- NULL
cancer_df_dc_copy$STDs.molluscum.contagiosum <- NULL
cancer_df_dc_copy$STDs.Hepatitis.B <- NULL

cancer_df_dc_copy$`STDs..Number.of.diagnosis`[cancer_df_dc_copy$`STDs..Number.of.diagnosis`==3] <- 2
cancer_df_dc_copy$`STDs..Number.of.diagnosis` <- as.factor(as.character(cancer_df_dc_copy$`STDs..Number.of.diagnosis`))

n <- nrow(cancer_df_dc)

start.time <- Sys.time()

#Fix a network
cancer_df_dc <- cancer_df_dc_copy
train_index <- (1:n)[-r]
test_index <- r

cancer_df_dc_test <-  cancer_df_dc[test_index,]
cancer_df_dc <- cancer_df_dc[train_index,]
#Random oversampling
n_minority <- nrow(cancer_df_dc_copy[cancer_df_dc_copy$Biopsy==1,])
n_majority <- nrow(cancer_df_dc_copy[cancer_df_dc_copy$Biopsy==0,])
oversampled_minority <- cancer_df_dc_copy[cancer_df_dc_copy$Biopsy==1,][sample(1:n_minority,n_majority,replace=TRUE),]
cancer_df_oversampled <- rbind(cancer_df_dc_copy[cancer_df_dc_copy$Biopsy==0,],oversampled_minority)

BN_score <- scoreparameters("bdecat",cancer_df_oversampled)
BN_order <- orderMCMC(BN_score,MAP=TRUE,scoreout=TRUE,chainout=TRUE)
temp_network <- empty.graph(rownames(BN_order$DAG))
amat(temp_network) <- BN_order$DAG

g1 <- graphviz.plot(temp_network)
graph::nodeRenderInfo(g1) <- list(fill="lightgreen", fontsize=50)
Rgraphviz::renderGraph(g1)

for(r in 1:nrow(cancer_df_dc_copy)){ #r<-7
  # str(cancer_df_dc)
  
  cancer_df_dc <- cancer_df_dc_copy
  train_index <- (1:n)[-r]
  test_index <- r
  
  cancer_df_dc_test <-  cancer_df_dc[test_index,]
  cancer_df_dc <- cancer_df_dc[train_index,]
  #Random oversampling
  n_minority <- nrow(cancer_df_dc[cancer_df_dc$Biopsy==1,])
  n_majority <- nrow(cancer_df_dc[cancer_df_dc$Biopsy==0,])
  oversampled_minority <- cancer_df_dc[cancer_df_dc$Biopsy==1,][sample(1:n_minority,n_majority,replace=TRUE),]
  cancer_df_oversampled <- rbind(cancer_df_dc[cancer_df_dc$Biopsy==0,],oversampled_minority)
  
  # table(cancer_df_oversampled$Biopsy) #balanced.
  
  # data(iris0)
  # str(iris0)
  #Fit BNs
  
  # cancer_df_dc <- cancer_df_oversampled
  
  
  
  # BN_score <- scoreparameters("bdecat",cancer_df_oversampled)
  # BN_order <- orderMCMC(BN_score,MAP=TRUE,scoreout=TRUE,chainout=TRUE)
  # temp_network <- empty.graph(rownames(BN_order$DAG))
  # amat(temp_network) <- BN_order$DAG
  # plot(BN_order$traceadd$orderscores)
  
  
  # fit_train <- bn.fit(temp_network,cancer_df_oversampled)
  fit_train <- bn.fit(temp_network,cancer_df_dc)

  #Method 1 starts
  temp_markov_blanket <- temp_network$nodes$Biopsy$mb
  # temp_markov_blanket <- names(cancer_df_dc_copy)[names(cancer_df_dc_copy)!="Biopsy"]
  # str2 = paste0(paste("`", temp_markov_blanket, "` == '",
  #              sapply(cancer_df_dc_test[1,temp_markov_blanket],as.character), "'", sep = ""),collapse="&")
  # 
  # str1 <- "(Biopsy==1)"
  # cmd = paste("cpquery(fit_train, ", str1, ", (",str2 , "),n=2000000)", sep = "")
  # prediction_vec[r] <- eval(parse(text = cmd))
  # cpquery(fit_train,(Biopsy==1),
  #         as.list(cancer_df_dc_test[,names(cancer_df_dc_test)!="Biopsy"]),method="lw",n=1000000)
  #Method 2 starts
  # prediction_vec[r] <- cpquery(fit_train,(Biopsy==1),
  #                              as.list(cancer_df_dc_test[,names(cancer_df_dc_test)!="Biopsy"]),method="lw",n=1000000)
  # 
  prediction_vec[r] <- cpquery(fit_train,(Biopsy==1),
                               as.list(cancer_df_dc_test[,temp_markov_blanket]),method="lw",n=1000000)
  
  #---------------
  
  print(paste0("Replication ",r))
  print(Sys.time()-start.time)
  temp_df <- data.frame(prediction_vec,cancer_df_dc_copy[1:r,"Biopsy"])
  print(t(temp_df))
}

#Plot a graph


#Analysis
# hist(prediction_vec)


#AUC
performance_df <- data.frame(array(dim=c(99,8)))
names(performance_df) <- c("c","TP","TN","FP","FN","Accuracy","TPR","FPR")
temp_counter <- 1
for(c in seq(0.01,0.99,by=0.01)){ #c<-0.07
  performance_df[temp_counter,"c"] <- c
  performance_df[temp_counter,"TP"] <- sum((prediction_vec>c)==1 & cancer_df_dc_copy$Biopsy==1,na.rm=TRUE)
  performance_df[temp_counter,"TN"] <- sum((prediction_vec>c)==0 & cancer_df_dc_copy$Biopsy==0,na.rm=TRUE)
  performance_df[temp_counter,"FP"] <- sum((prediction_vec>c)==1 & cancer_df_dc_copy$Biopsy==0,na.rm=TRUE)
  performance_df[temp_counter,"FN"] <- sum((prediction_vec>c)==0 & cancer_df_dc_copy$Biopsy==1,na.rm=TRUE)
  performance_df[temp_counter,"Accuracy"] <- sum(as.numeric(prediction_vec>c)==cancer_df_dc_copy$Biopsy,na.rm=TRUE)/
    sum(!is.na(prediction_vec))
  performance_df[temp_counter,"TPR"] <-performance_df[temp_counter,"TP"]/(performance_df[temp_counter,"TP"] +performance_df[temp_counter,"FN"] )
  performance_df[temp_counter,"FPR"] <-performance_df[temp_counter,"FP"]/(performance_df[temp_counter,"FP"] +performance_df[temp_counter,"TN"] )
  temp_counter <- temp_counter+1
}

# hist(prediction_vec)
#1
plot(performance_df$c,performance_df$Accuracy,type="o")

plot(performance_df$c,performance_df$TPR,type="o")
lines(performance_df$c,1-performance_df$FPR,type="o")


#2. AUC
plot(performance_df$FPR,performance_df$TPR,type="o")
abline(a=0,b=1)

store_df[[seed]] <- performance_df
}


# install.packages("cvAUC")
library(cvAUC)

