setwd(dirname(rstudioapi::getSourceEditorContext()$path))
cancer_df <- read.csv("risk_factors_cervical_cancer.csv",check.names=FALSE)




# table(cancer_df$`Number of sexual partners`)
# hist()
library(BiDAG)
# install.packages("arules")
library(arules)


#Imputation
cancer_df[cancer_df=="?"] <- NA
str(cancer_df)

#check missing
temp_missing <- apply(cancer_df,2,function(x) sum(is.na(x)))/nrow(cancer_df)*100
temp_missing <- data.frame(temp_missing)
rownames(temp_missing)
for(c in 1:ncol(cancer_df)){
  cancer_df[,c] <- as.numeric(cancer_df[,c])
}
#Drop the first time diagonsed
cancer_df$`STDs: Time since first diagnosis` <- NULL
cancer_df$`STDs: Time since last diagnosis`  <- NULL

sum(!complete.cases(cancer_df[cancer_df$Biopsy==1,])) #we can drop directly
nrow(cancer_df[cancer_df$Biopsy==1,])

sum(!complete.cases(cancer_df))
nrow(cancer_df)

#Drop unrelated columns: Dropped two X's and three Y's
lapply(cancer_df,unique)
cancer_df$`STDs:cervical condylomatosis` <- NULL #All are zeros
cancer_df$`STDs:AIDS` <- NULL #All are zeros

cancer_df$Citology <- NULL #not used. These are target variables.
cancer_df$Schiller <- NULL
cancer_df$Hinselmann <- NULL

#Impute by mode
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

for(c in 1:ncol(cancer_df)){
  if(sum(is.na(cancer_df[,c]))>0){
    cancer_df[,c][is.na(cancer_df[,c])] <- getmode(cancer_df[,c])
  }
}

#check complete cases
cancer_df[!complete.cases(cancer_df),] #no missing values now


#Discretization
#prepare a dataframe for storing new variables
cancer_df_dc <- cancer_df
unique_val <- lapply(cancer_df,unique)
library(psych)
library(gtools)
pairs.panels(cancer_df[,c(1,2,3,4,6,7,9,11)])

#quantile cut

# for(c in c(1,2,3,4,6,7,9,11)){ #c<-2
#   if(c!=2){
#   cancer_df_dc[,c] <- quantcut(cancer_df[,c],q=8)
#   }else{
#     quantcut(cancer_df[,c],q=4)
#     #1, 2, 3, [4,7], >7
#     cancer_df_dc[,c] <- (cancer_df[,c]==1)*1+(cancer_df[,c]==2)*2+
#       (cancer_df[,c]==3)*3+(cancer_df[,c]>=4 & cancer_df[,c]<=7)*4+(cancer_df[,c]>7)*5
#     cancer_df_dc[,c] <- as.factor(c("1", "2", "3", "[4,7]", ">7")[cancer_df_dc[,c]])
#   }
# }

names(cancer_df_dc)[c(1,2,3,4,6,7,9,11)]
lapply(cancer_df_dc[,c(1,2,3,4,6,7,9,11)],table)

cancer_df_dc$Age <- cut(cancer_df_dc$Age,breaks=c(0,10,15,20,25,30,35,40,45,50,70,90))
cancer_df_dc$`Number of sexual partners` <- cut(cancer_df_dc$`Number of sexual partners`,breaks=c(0,1,2,3,4,5,6,8,28))
cancer_df_dc$`First sexual intercourse` <- cut(cancer_df_dc$`First sexual intercourse`,breaks=c(0,12,14,15,16,17,18,20,22,24,26,28,32))
cancer_df_dc$`Num of pregnancies` <- cut(cancer_df_dc$`Num of pregnancies`,breaks=c(-1,0,1,2,3,4,5,6,7,8,11))
cancer_df_dc$`Smokes (years)` <- cut(cancer_df_dc$`Smokes (years)`,breaks=c(-1,0,1,2,3,4,5,7,9,10,13,15,20,24,28,37))
cancer_df_dc$`Smokes (packs/year)` <- cut(cancer_df_dc$`Smokes (packs/year)`,breaks=c(-1,0,1,2,3,4,5,6,8,15,37))
cancer_df_dc$`Hormonal Contraceptives (years)` <- cut(cancer_df_dc$`Hormonal Contraceptives (years)`,breaks=c(-1,0,1,2,3,4,5,6,7,8,9,10,12,14,16,20,30))
cancer_df_dc$`IUD (years)` <- cut(cancer_df_dc$`IUD (years)`,breaks=c(-1,0,1,2,3,5,7,9,12,19))

lapply(cancer_df_dc,unique)

for(c in 1:ncol(cancer_df_dc)){
  cancer_df_dc[,c] <- as.factor(cancer_df_dc[,c])
  cancer_df[,c] <- as.factor(cancer_df[,c])
}

str(cancer_df_dc)


# table(cancer_df_dc$`Number of sexual partners`)
write.csv(cancer_df,"cancer_data_processed.csv",row.names=FALSE)
write.csv(cancer_df_dc,"cancer_data_discretized.csv",row.names=FALSE)

apply(cancer_df_dc,2,table)


#oversampling
# cancer_df_dc_oversampled <- mwmote(cancer_df, numInstances = 100,classAttr="Biopsy")


# write.csv(cancer_df,"cancer_data_processed.csv",row.names=FALSE)
# write.csv(cancer_df_dc,"cancer_data_discretized.csv",row.names=FALSE)