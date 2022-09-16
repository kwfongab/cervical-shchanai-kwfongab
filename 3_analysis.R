seed_vec <- c(5,7,9,10,13,14,16,17,18,20)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
# load("50_replications.RData")

result_df <- data.frame(array(dim=c(101*20,4)))
auc_df <- vector()
performance_df <- data.frame()
names(result_df) <- c("c","TPR","FPR","seed")
i <- 1
check_normal_case <- vector()
for(s in 1:20){ #s<-1
  check_normal_case[s] <- !all(attr(store_output[[s]][[1]],"prob")[2,]==0.5)
  # store_output[[s]]
  #Calculate AUC
  for(c in seq(0,100,by=1)){ #c<-1
    temp_pred <-  (attr(store_output[[s]][[1]],"prob")[2,]>=c/100)*1
    TP <- sum(temp_pred==1&store_output[[s]][[2]]==1,na.rm=TRUE)
    TN <- sum(temp_pred==0&store_output[[s]][[2]]==0,na.rm=TRUE)
    FP <- sum(temp_pred==1&store_output[[s]][[2]]==0,na.rm=TRUE)
    FN <- sum(temp_pred==0&store_output[[s]][[2]]==1,na.rm=TRUE)
    result_df[i,"c"] <- c/100
    result_df[i,"TPR"] <- TP/(TP+FN)
    result_df[i,"FPR"] <- FP/(FP+TN)
    result_df[i,"seed"] <- s
  
    i <- i+1
  }
   
  plot(result_df[result_df$seed==s,"FPR"],result_df[result_df$seed==s,"TPR"],type="o")
  abline(a=0,b=1)
  require(DescTools)

}

for(s in 1:20){
  auc_df[s] <- DescTools::AUC(rev(result_df[result_df$seed==s,"FPR"]),rev(result_df[result_df$seed==s,"TPR"]))
}

boxplot(auc_df,ylab="AUC")
grid()

library(ggplot2)
result_df$seed <- as.factor(result_df$seed)
ggplot(result_df,aes(x=FPR,y=TPR,color=seed))+geom_point()+
  geom_line()+theme_light()+geom_abline(slope=1,intercept=0)+ 
  theme(legend.title = element_blank(),legend.position = "none")+
  xlab("1-FPR")

#Pick the one maximizing the AUC.

s <- which.max(auc_df)

# saveRDS(store_output,"prediction_output.RDS")
# check_normal_case[s] <- !all(attr(store_output[[s]][[1]],"prob")[2,]==0.5)
# store_output[[s]]
#Calculate AUC

train_test_df <- data.frame()
store_output[[s]]
for(c in 0:100){
  #Training
  train_test_df[c+1,"train_acc"] <- mean((attr(store_output[[s]][[3]],"prob")[2,]>(c/100))*1==store_output[[s]][[4]],na.rm=TRUE)
  temp_pred <- (attr(store_output[[s]][[3]],"prob")[2,]>(c/100))*1
  TP <- sum(temp_pred==1&store_output[[s]][[4]]==1,na.rm=TRUE)
  TN <- sum(temp_pred==0&store_output[[s]][[4]]==0,na.rm=TRUE)
  FP <- sum(temp_pred==1&store_output[[s]][[4]]==0,na.rm=TRUE)
  FN <- sum(temp_pred==0&store_output[[s]][[4]]==1,na.rm=TRUE)
  
  train_test_df[c+1,"train_TPR"] <- TP/(TP+FN)
  train_test_df[c+1,"train_Specificity"] <- 1-FP/(FP+TN)
  #Testing
  temp_pred <- (attr(store_output[[s]][[1]],"prob")[2,]>(c/100))*1
  TP <- sum(temp_pred==1&store_output[[s]][[2]]==1,na.rm=TRUE)
  TN <- sum(temp_pred==0&store_output[[s]][[2]]==0,na.rm=TRUE)
  FP <- sum(temp_pred==1&store_output[[s]][[2]]==0,na.rm=TRUE)
  FN <- sum(temp_pred==0&store_output[[s]][[2]]==1,na.rm=TRUE)
  
  train_test_df[c+1,"test_TPR"] <- TP/(TP+FN)
  train_test_df[c+1,"test_Specificity"] <- 1-FP/(FP+TN)
  train_test_df[c+1,"test_acc"] <- mean((attr(store_output[[s]][[1]],"prob")[2,]>(c/100))*1==store_output[[s]][[2]],na.rm=TRUE)
  train_test_df[c+1,"c"] <- c/100
}

train_test_df_melt <- melt(train_test_df,id="c")

ggplot(train_test_df_melt[train_test_df_melt$variable %in% 
                            c("train_TPR","test_TPR","train_Specificity","test_Specificity"),],
       aes(x=c,y=value,color=variable)) + geom_point()+geom_line()+theme_light()+
  xlab("cut-off")+ylab("TPR/Specificity")

#choosing c=0.5 is the best.
c <- 0.5
temp_pred <- (attr(store_output[[s]][[1]],"prob")[2,]>c)*1
TP <- sum(temp_pred==1&store_output[[s]][[2]]==1,na.rm=TRUE)
TN <- sum(temp_pred==0&store_output[[s]][[2]]==0,na.rm=TRUE)
FP <- sum(temp_pred==1&store_output[[s]][[2]]==0,na.rm=TRUE)
FN <- sum(temp_pred==0&store_output[[s]][[2]]==1,na.rm=TRUE)

(TP+TN)/(TP+TN+FP+FN) #Accuracy 77.3%
TPR<- TP/(TP+FN) #True positive rate 81.8%
1-FP/(FP+TN) #Specificity=81.8%
Precision <- TP/(TP+FP) #Precision: 72.1%
2*TPR*Precision/(TPR+Precision) #F-measure 19.4%

#Convergence plot

BDeu_df <- data.frame()

# for(m in 1:20){
#   BDeu_df[((m-1)*1000+1):(1000*m),"seed"] <- m
#   BDeu_df[((m-1)*1000+1):(1000*m),"Iteration"] <- 1:1000
#   BDeu_df[((m-1)*1000+1):(1000*m),"BDeu"] <- store_output[[m]][[5]][1:1000]
# }
# 
# 
# 
# ggplot(BDeu_df,aes(x=Iteration,y=BDeu,color=seed))+geom_line()+
#   theme_light()+xlab("Iteration")+ylab("BDeu score")

#Using all replications to find the performances

performances_df <- data.frame()

for(s in 1:20){
  c <- 0.5
  temp_pred <- (attr(store_output[[s]][[1]],"prob")[2,]>c)*1
  TP <- sum(temp_pred==1&store_output[[s]][[2]]==1,na.rm=TRUE)
  TN <- sum(temp_pred==0&store_output[[s]][[2]]==0,na.rm=TRUE)
  FP <- sum(temp_pred==1&store_output[[s]][[2]]==0,na.rm=TRUE)
  FN <- sum(temp_pred==0&store_output[[s]][[2]]==1,na.rm=TRUE)
  
  performances_df[s,"AUC"] <- 
  performances_df[s,"Accuracy"] <- (TP+TN)/(TP+TN+FP+FN) #Accuracy 77.3%
  performances_df[s,"Sensitivity"] <- TP/(TP+FN) #True positive rate 81.8%
  performances_df[s,"Specificity"] <- 1-FP/(FP+TN) #Specificity=81.8%
  performances_df[s,"Precision"] <- TP/(TP+FP) #Precision: 72.1%
  performances_df[s,"F-measure"] <- 2*performances_df[s,"Sensitivity"]*performances_df[s,"Precision"]/(performances_df[s,"Sensitivity"]+performances_df[s,"Precision"]) #F-measure 19.4% 
}

performance_df2 <- cbind(auc_df,performances_df)
names(performance_df2)[1] <- "AUC"
# boxplot(performance_df2)
apply(performance_df2,2,median)
grid()

apply(cancer_df_dc,2,table)

graphviz.plot()
