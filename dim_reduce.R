library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(nnet)
library(randomForest)
library(xgboost)
library(e1071)
train_all <- read_csv('Downloads/Jdata/train_all.csv')
train_all[is.na(train_all)] <- 0
########## SVD ##################################
data <- train_all[-length(train_all)]

# very slow
mysvd <- function(data, precision=0.90){
  svd <- svd(data)
  singular <- svd$d
  U <- svd$u
  V <- svd$v
  i = 1
  while (sum(singular[1:i])/sum(singular) < precision){ 
    i=i+1
  }
  recon <- U[ ,1:i, drop=F]%*%diag(singular[1:i], i)%*%t(V[ ,1:i, drop=F])
  error <- sum((recon-data)^2)
  return (list(num=i, recon=recon, error=error))
}

########## PCA ##################################
princomp(data)
prcomp(data) # use svd method 

mypca <- function(data, cor=FALSE, precision=0.90){
  if (cor==FALSE) mat <- cov(data)
  if (cor==TRUE) mat <- cor(data)
  eig <- eigen(mat)
  i = 1
  while (sum(eig$values[1:i])/sum(eig$values) < precision){ 
    i = i+1
  }
  acc = sum(eig$values[1:i])/sum(eig$values)
  recon <- scale(data)%*%eig$vectors[, 1:i] # compute the scale score
  return(list(num=i, recon=recon, accumulate=acc))
}
