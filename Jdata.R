library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(gbm)
library(nnet)
library(neuralnet)
library(randomForest)
library(xgboost)
library(e1071)
library(h2o)
library(pROC)
##########################################################
############# import data ################################
##########################################################

# Comment <- read.csv('Downloads/JData/JData_Comment.csv',colClasses = c(rep('factor',4),'numeric'))
# Users <- read.csv('Downloads/JData/JData_User.csv',colClasses = 'factor',fileEncoding='GBK')
# change the time zone and the coding
set_locale = locale(tz=Sys.timezone(), encoding='GBK')
Product <- read_csv('Downloads/JData/JData_Product.csv')
Comment <- read_csv('Downloads/JData/JData_Comment.csv')
# take the newest weekly accumualted comments
Comment_new <- filter(Comment, dt=='2016-04-15')[-1]
Users <- read_csv('Downloads/JData/JData_User.csv', locale=set_locale)
Action_02 <- read_csv('Downloads/JData/JData_Action_201602.csv', locale=set_locale, cols(user_id=col_integer()))
Action_03 <- read_csv('Downloads/JData/JData_Action_201603.csv', locale=set_locale, cols(user_id=col_integer()))
Action_04 <- read_csv('Downloads/JData/JData_Action_201604.csv', locale=set_locale, cols(user_id=col_integer()))
tbl_02 <- left_join(left_join(Action_02, Users, by="user_id"), Comment_new, by='sku_id') 
tbl_03 <- left_join(left_join(Action_03, Users, by="user_id"), Comment_new, by='sku_id')
tbl_04 <- left_join(left_join(Action_04, Users, by="user_id"), Comment_new, by='sku_id')
tbl_all <- bind_rows(tbl_02, tbl_03, tbl_04) # join all tables together

######## objection subset ###########################
tbl_cate8 <- filter(tbl_all, cate==8)[-6] # we chose only cate=8 rows, delete the col cate
write_csv(tbl_cate8, 'Downloads/JData/tbl_cate8.csv') 
tbl_cate8 <- read_csv('Downloads/JData/tbl_cate8.csv', locale=locale(tz=Sys.timezone()), col_types = cols(user_id='i')) 

#######################################################
########## data cleaning ##############################
#######################################################

####### crosstable of {user, sku, brand} with type
cross_table <- function(data, id=NULL){
data %>%
  group_by_(id, ~type) %>%
  summarise( count=n()) %>%
  spread(type, count, fill=0) 
}
####### crosstable 'alternative version'
# user_type <- table(tbl_clean$user_id, tbl_clean$type)
# addmargins(users_type)

data_clear <- function(data) {
 # remove sku_id where type{2, 4 , 5}==0, namely: 'bad sku' 
 sku_type <- cross_table(tbl_cate8, 'sku_id')
 sku_keep <- sku_type[!(sku_type$`2`==0 & sku_type$`4`==0 & sku_type$`5`==0), 1]
 # remove user_id where type{2, 4 ,5}==0 and type{1, 6}<C, namely: 'least potential buyer' 
 user_type <- cross_table(tbl_cate8, 'user_id')
 user_keep <- user_type[!(user_type$`6`<=17&user_type$`1`<=26&user_type$`4`==0&user_type$`2`==0&user_type$`5`==0), 1]
 # turn brand, age, sex into dummy variable
 brand_vector <- tbl_df(class.ind(tbl_cate8$brand))
 age_vector <- tbl_df(class.ind(tbl_cate8$age))
 sex_vector <- tbl_df(class.ind(tbl_cate8$sex))
 colnames(sex_vector) <- c('M', 'F', 'unknown')
 tbl_dummy <- select(bind_cols(tbl_cate8, age_vector, sex_vector, brand_vector), -age, -sex)
 # turn date into days 
 temp <- mutate(tbl_dummy, user_reg_tm=as.integer(as.Date('2016-04-15')-user_reg_tm))
 # delete the useless user and sku
 tbl_clean <- inner_join(inner_join(temp, sku_keep), user_keep)
 return(tbl_clean)
}
tbl_clean <- data_clear(tbl_cate8)

#######################################################
############# Data Anlysis ############################
#######################################################

###### visualization
# ggplot(user_count, aes(x=count, y=..density..)) + geom_density(alpha=0.3) + facet_grid(.~type)
ggplot(user_type, aes(x=`1`, y=..density..)) + geom_density(alpha=0.3)
ggplot(user_type, aes(x=`4`)) + geom_histogram(alpha=0.3, binwidth=0.5)
ggplot(user_type, aes(x=`6`)) + geom_histogram(alpha=0.8, binwidth=5)
# data is too skewed to show
# ggplot(user_count, aes(x=type, y=count, group=type)) + geom_boxplot()
ggplot(user_count, aes(x=1, y=count)) + geom_boxplot()

###### users clustering 
options(digits = 6)
u <- left_join(user_type, Users, by='user_id')
age <- tbl_df(class.ind(u$age))
a1 <- select(u, -age, -user_reg_tm) 
a2 <- tbl_df(scale(a1[-1]))
u <-  na.omit(bind_cols(a2, age))
u_clu <- kmeans(u, centers=4)

###### users actions tracking
user_tracking <- function(data, ...){
  return(filter(data, ...))
}

###### top type action
top_type <- function(data, id=NULL, type_val=4){
  'id is {user_id, sku_id, brand} must be quoted like ~, quote(), ""
  type_val: 1, 2, 3, 4, 5, 6; defulat value is 4'
  data %>%
  filter_(~type==type_val) %>% # SE version
  group_by_(id) %>%
  summarise(count=n()) %>%
  arrange(desc(count)) -> data
  View(data)
  return(data)
}

###### prior info
tbl_clean %>%
  filter(time>'2016-04-10' & time<'2016-04-16' & type==2 ) %>%
  group_by(user_id, sku_id) %>%
  filter(type!=4) %>%
  summarise(count=n()) %>%
  arrange(desc(count)) -> potential_last5
write_csv(potential_last5, 'Downloads/JData/potential_last5.csv')

#########################################################
########### Feature Extraction ##########################
#########################################################
user_turning_rate <- mutate(user_type, u_sum=`1`+`2`+`3`+`4`+`5`+`6`,
                             u1=`4`/(`1`+`6`), u2=`2`/(`1`+`6`), u3=`5`/(`1`+`6`), u4=`4`/(`2`+`5`))

sku_turning_rate <- mutate(sku_type, s_sum=`1`+`2`+`3`+`4`+`5`+`6`,
                           s1=`4`/(`1`+`6`), s2=`2`/(`1`+`6`), s3=`5`/(`1`+`6`), s4=`4`/(`2`+`5`))

brand_turning_rate <- mutate(brand_type, b_sum=`1`+`2`+`3`+`4`+`5`+`6`,
                             b1=`4`/(`1`+`6`), b2=`2`/(`1`+`6`), b3=`5`/(`1`+`6`), b4=`4`/(`2`+`5`))

###### chose last1, 3, 7 days for more features
latest_1 <- function(data){
  filter(data, time>time[length(time)]-24*60*60)
}
latest_3 <- function(data){
  filter(data, time>time[length(time)]-3*24*60*60)
}
latest_7 <- function(data){
  filter(data, time>time[length(time)]-7*24*60*60)
}

####### add features to user_id, sku_id, brand
feature <- function(data, id){
  'id: user_id, sku_id or brand, must be quoted
  turning rate: r1=4/(1+6), r2=2/(1+6), r3=5/(1+6), r4=4/(2+5) '
  data %>%
    group_by_(id, 'type') %>%
    summarise(count=n()) %>%
    spread(type, count, fill=0) %>% 
    mutate(sum=`1`+`2`+`3`+`4`+`5`+`6`,
           r1=(`4`+0.01)/(`1`+`6`+0.01), r2=(`2`+0.01)/(`1`+`6`+0.01), # to avoid Na, NaN, Inf
           r3=(`5`+0.01)/(`1`+`6`+0.01), r4=(`4`+0.01)/(`2`+`5`+0.01)) -> data 
    colnames(data)[-1] <- paste(substr(id, 1, 1), '_',colnames(data)[-1], sep='') # chage the colnames
    return(data)
}

####### join the 1, 3, 7 days features together
feature_join <- function(data, id){
  l_1 <- latest_1(data)
  l_3 <- latest_3(data)
  l_7 <- latest_7(data)
  f_1 <- feature(l_1, id)
  f_3 <- feature(l_3, id)
  f_7 <- feature(l_7, id)
  # by = deparse(substitute(...))
  # be careful use f_7 as the left side  suffix's param cannot be '', it cause fatal error!!!!
  obj_feature <- left_join(left_join(f_7, f_3, by=id, suffix=c('_l7', '_l3')), f_1, by=id)
  return(obj_feature)
}
####### add {user, sku, brand} feature to the trainset
trainset_with_feature <- function(data){
  user_feature <- feature_join(data, 'user_id')
  sku_feature <- feature_join(data, 'sku_id')
  brand_feature <- feature_join(data, 'brand')
  left_join(left_join(left_join(data, user_feature, by='user_id'),
                      sku_feature, by='sku_id'), brand_feature, by='brand')
}

#############################################################
########### Data Preprocessing  #############################
#############################################################

###### replacing Na with zero
# method 1 Na and NaN can be checked by is.na(), but Inf can not, do not work??
trainset <- trainset_1[is.na(trainset_1)] <- 0
# method 2
trainset_1 %>%
  replace(., is.na(.), 0) -> trainset
# method 3 fast
trainset_1 %>%
  mutate_all(funs(replace(., is.na(.), 0))) -> trainset
# method 4 fast, but sometimes do not work??
trainset2 <- replace_na(trainset_1, list(0))

######### efficiency of several replace NA methods #########
library(microbenchmark)
# Numerics
set.seed(24)
dfN <- as.data.frame(matrix(sample(c(NA, 1:5), 1e6 *12, replace=TRUE),
                            dimnames = list(NULL, paste0("var", 1:12)), 
                            ncol=12))

opN <- microbenchmark(
  baseR_replace    = local(dfN %>% replace(., is.na(.), 0)),
  subsetReassign   = local(dfN[is.na(dfN)] <- 0),
  mut_at_replace   = local(dfN %>% mutate_at(funs(replace(., is.na(.), 0)), .cols = c(1:12))),
  mut_all_replace  = local(dfN %>% mutate_all(funs(replace(., is.na(.), 0)))),
  replace_na       = local(dfN %>% replace_na(list(var1 = 0, var2 = 0, var3 = 0, var4 = 0, var5 = 0, var6 = 0, var7 = 0, var8 = 0, var9 = 0, var10 = 0, var11 = 0, var12 = 0))),
  times = 1000L
)
#############################################################
normalization <- function(data){
  center <- sweep(data, 2, apply(data, 2, min),'-') #在列的方向上减去最小值，不加‘-’也行
  R <- apply(data, 2, max) - apply(data, 2, min) #算出极差，即列上的最大值-最小值
  data_nor<- sweep(center, 2, R, "/") #把减去均值后的矩阵在列的方向上除以极差向量
  return(data_nor)
}
trainset_nor <- normalization(replace_na(trainset_1, list(0))) 

####### create labels
create_label <- function(data, type=1){
  ' type: 1 means label 1 and 0; 2 means label 1 and -1'
  if (type == 1){
    labels <- ifelse(data$type==4, 1, 0)
  }
  else if (type ==2){
    labels <- ifelse(data$type==4, 1, -1)
  }
  else {stop("wrong type")}
  df <- cbind(data[1], labels)
  return(distinct(tbl_df(df)))
}

###### create trainset and testset
create_trainset <- function(from='2016-02-01' ,last=7){
  from <- as.POSIXct(from)
  train <- filter(tbl_clean, time>from & time<(from+last*24*60*60))
  label <- filter(tbl_clean, time>(from+last*24*60*60) & time<(from+(last+5)*24*60*60))
  train_f <- distinct(trainset_with_feature(train)[-c(3:5)]) # del time, model_id, type
  if (nrow(label)!=0){
    trainset <- left_join(train_f, create_label(label))[-c(1, 2, 3)] # del user_id, sku_id, brand
    trainset_scale <- mutate(tbl_df(scale(trainset)), labels=ifelse(labels>0, 1, 0)) # labels have NA, u, brand have NA, NaN
  }
  else { 
    id <- select(train_f, user_id, sku_id)
    body <- train_f[-c(1, 2, 3)]
    scale <- tbl_df(scale(body))
    bind <- bind_cols(id, scale)
    trainset_scale <- filter(bind, u_4_l7<0) # important, delete the user-sku pair that already bought
  }
  return(trainset_scale)
}

train_1 <- create_trainset('2016-02-01')
train_2 <- create_trainset('2016-02-08')
train_3 <- create_trainset('2016-02-15')
train_4 <- create_trainset('2016-02-22')
train_5 <- create_trainset('2016-03-01')
train_6 <- create_trainset('2016-03-09')
train_7 <- create_trainset('2016-03-14')
train_8 <- create_trainset('2016-03-21')
train_9 <- create_trainset('2016-03-28')
train_10 <- create_trainset('2016-04-04')
test <- create_trainset('2016-04-09')

test[is.na(test)] <- 0
anyNA(test)
write_csv(test, 'Downloads/JData/test.csv')
test <- read_csv('Downloads/JData/test.csv')
test_id <- test[ ,1:2]
test_model <- test[ ,-(1:2)]

############## imbalance ##########################
# ratio of labels ==1 
ratio <- function(data){
  sum(data$labels==1, na.rm=T)/nrow(data)
}

# train_oversampling
train_oversampling <- function(data){
  data %>%
    filter(labels==0 | is.na(labels)) %>%
    sample_frac(0.9) -> sample
  data <- setdiff(data, sample) 
  return(data)
}

train1 <- train_oversampling(train_1)
train2 <- train_oversampling(train_2)
train3 <- train_oversampling(train_3)
train4 <- train_oversampling(train_4)
train5 <- train_oversampling(train_5)
train6 <- train_oversampling(train_6)
train7 <- train_oversampling(train_7)
train8 <- train_oversampling(train_8)
train9 <- train_oversampling(train_9)
train10 <- train_oversampling(train_10)

train_all <- bind_rows(train1, train2, train3, train4, train5, 
                       train6, train7, train8, train9, train10)
train_all[is.na(train_all)] <- 0 # replace NA together
anyNA(train_all)
write_csv(train_all, 'Downloads/JData/train_all.csv')
train_all <- read_csv('Downloads/JData/train_all.csv')

# divide training and testing set
set.seed(12345)  
sub <- sample(1:nrow(train_all),round(nrow(train_all)/4))  
trainset <- train_all[-sub, ]
testset <- train_all[sub, ]

####################################################
############### MODEL ONE ##########################
########## LOGISTIC REGRESSION #####################
####################################################
train_labels <- select(trainset, labels)
a1 <- princomp(trainset[-length(trainset)])
train_feature <- tbl_df(a1$score[ ,1:56])
trainset_pca <- bind_cols(train_feature, train_labels)

test_labels <- select(testset, labels)
a2 <- princomp(testset[-length(testset)])
test_feature <- tbl_df(a2$score[ ,1:56])
testset_pca <- bind_cols(test_feature, test_labels)


logit_fit1 <- glm(labels~., family=binomial(link='logit'), data=trainset, maxit=10000) # Z-score scale is much better, with oversampling
logit_fit2 <- glm(labels~., family=binomial(link='logit'), data=trainset_pca, control=list(maxit=10000) ) # use PCA, improve training error but not test
# link <- predict(logit_fit, select(trainset_0213_0219_nor, -labels)) # equals to ln(p/(1-p))，threshold is 0
# p <- exp(link)/(1+exp(link))

LR <- function(logit_fit, trainset, testset){
  # trainset evaluation
  train_predict <- ifelse(predict(logit_fit) > 0, 1, 0)
  table1 <- table(trainset$labels, train_predict) 
  # predict the testset, type = response obtain the prob directly
  p <- predict(logit_fit, select(testset, -labels), type='response') # threshold is 0.5
  test_predict <- ifelse(p > 0.5, 1, 0)
  table2 <- table(testset$labels, test_predict)
  # compute precision, recall and F-measure
  precision <- sum(testset$labels & test_predict)/sum(test_predict==1)
  recall <- sum(testset$labels & test_predict)/sum(testset$labels==1)
  F <- 2*precision*recall/(precision+recall) 
  accuracy <- c(precision=precision, recall=recall, F_measure=F) # testset
  summary <- summary(logit_fit)
  result <- list(crosstable_train=table1, crosstable_test=table2, accuracy=accuracy, summary=summary)
  return(result)
}

########## ROC CURVE ############################################
roc_curve <- function(predict_value, true_value){
  data <- data.frame(prob=predict_value, obs=true_value)
  data <- data[order(data$prob), ] # 将预测概率按照从低到高排序
  n <- nrow(data)
  tpr <- fpr <- rep(0, n)
  for (i in 1:n){
    threshold <- data$prob[i] # 根据不同的threshold来计算TPR和FPR
    tp <- sum(data$prob>threshold & data$obs==1)
    fp <- sum(data$prob>threshold & data$obs==0)
    tn <- sum(data$prob<threshold & data$obs==0)
    fn <- sum(data$prob<threshold & data$obs==1)
    tpr[i] <- tp/(tp+fn)  # 真正率
    fpr[i] <- fp/(tn+fp)  # 假正率
  }
  ggplot(data, aes(x=fpr, y=tpr)) + geom_line() + geom_abline(slope=1, intercept=0, color='red')
}

###### use pROC package
modelroc <- roc(trainset_0201_0207_nor$labels, predict(logit_fit))
plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1,0.2), 
     grid.col=c("green","red"), max.auc.polygon=TRUE, auc.polygon.col="blue", print.thres=TRUE)

####### get the predict user_id and sku_id ###################################
prob <- predict(logit_fit1, as.data.frame(test_model), type='response')

# choose all the user-sku pairs
test %>%
  mutate(prob) %>%
  filter(prob>0.5) %>%
  select(user_id, sku_id, prob) %>% 
  arrange(desc(prob)) -> predict_alllist

# choose the best prob user-sku pairs
test %>%
  mutate(prob) %>%
  filter(prob>0.5) %>%
  select(user_id, sku_id, prob) %>% 
  arrange(desc(prob)) %>%
  group_by(user_id) %>%
  slice(1) %>% # chose the highest prob row for each user_id
  arrange(desc(prob)) -> predict_list 
write_csv(predict_alllist, 'Downloads/JData/result_LR_all.csv') # 3cols
write_csv(predict_list, 'Downloads/JData/result_LR.csv') # 3cols

######################################################
################# MODEL TWO ##########################
################ Random Forest #######################
######################################################
# rf method needs large dataset, and slow, pca is much worse, 1~2 hours 
set.seed(1234)
####### make small sampleset to tune param ################################
tt <- sample_frac(trainset, 0.1)
ts <- sample_frac(testset, 0.1)
aa <- sample_frac(trainset_pca, 0.05)
as <- sample_frac(testset_pca, 0.2)

rf1 <- randomForest(x=select(tt, -labels), y=as.factor(tt$labels),
                    xtest=select(ts, -labels), ytest=as.factor(ts$labels),
                    ntree=300, importance=T, mtry=30) 
rf2 <- randomForest(x=select(tt, -labels), y=as.factor(tt$labels),
                    xtest=select(ts, -labels), ytest=as.factor(ts$labels), mtree=200)
####################### Model Usage ###############################################
## S3 method for class 'formula' 
randomForest(formula, data=NULL, ..., subset, na.action=na.fail)
## Default S3 method: key parameters: mtry, ntree, nodesize
randomForest(x, y=NULL,  xtest=NULL, ytest=NULL, ntree=500,
             mtry=if (!is.null(y) && !is.factor(y))
               max(floor(ncol(x)/3), 1) else floor(sqrt(ncol(x))),
             replace=TRUE, classwt=NULL, cutoff, strata,
             sampsize = if (replace) nrow(x) else ceiling(.632*nrow(x)),
             nodesize = if (!is.null(y) && !is.factor(y)) 5 else 1,
             maxnodes = NULL,
             importance=FALSE, localImp=FALSE, nPerm=1,
             proximity, oob.prox=proximity,
             norm.votes=TRUE, do.trace=FALSE,
             keep.forest=!is.null(y) && is.null(xtest), corr.bias=FALSE,
             keep.inbag=FALSE, ...)

############################## My Model ###################################
# mtry: number of features randomly sampled; samplesize: snumber of rows
# ntree: number of tree to grow; nodesize, maxnodes
rf1 <- randomForest(x=select(trainset, -labels), y=as.factor(trainset$labels),
                    xtest=select(testset, -labels), ytest=as.factor(testset$labels),
                    ntree=100, importance=T, proximity=F) 
rf2 <- randomForest(x=select(trainset, -labels), y=as.factor(trainset$labels),
                    xtest=select(testset, -labels), ytest=as.factor(testset$labels), 
                    ntree=100, importance=T, nodesize=50) # defaulted nodesize=1 is much better than nodesize=50
rf3 <- randomForest(x=select(trainset, -labels), y=as.factor(trainset$labels),
                    #xtest=select(testset, -labels), ytest=as.factor(testset$labels), 
                    ntree=100, importance=T, mtry=20) # mtry=sqrt(m)=12 is better than mtry=6， mtry=20 is better
rf4 <- randomForest(x=select(trainset, -labels), y=as.factor(trainset$labels),
                    xtest=select(testset, -labels), ytest=as.factor(testset$labels), 
                    ntree=300, importance=T, mtry=20, keep.forest=T)

####### model components
rf1$votes
rf1$confusion
rf1$test
plot(rf4) # decide the ntree
pred <- predict(rf4, test_model, type='response', norm.votes=T) # type: response, prob, vote
pred <- rf4$test$predicted
prob <- rf4$test$votes[ ,2]

F_measure <- function(model){
  precision <- model$test$confusion[4]/sum(model$test$confusion[3:4])
  recall <- model$test$confusion[4]/sum(model$test$confusion[c(2,4)])
  F <- 2*precision*recall/(precision+recall) 
  return(list(precision=precision, recall=recall, F_measure=F))
}

# F_measure=0.724

importance(rf4,type=1)  # 重要性评分  
importance(rf4,type=2)  # Gini指数  
varImpPlot(rf4)    

###### result 
test %>%
  select(user_id, sku_id) %>% 
  mutate(prob) %>% 
  arrange(desc(prob)) -> predict_alllist

rf_p <- test[,1:2][list(pred==1), ]

write_csv(predict_alllist, 'Downloads/JData/rf_all.csv')
write_csv(rf_p, 'Downloads/JData/rf_pred.csv')

####################################################
############### MODEL THREE #########################
################ SVM ###############################
####################################################
svmfit1 <- svm(x=trainset[-ncol(trainset)], y=trainset$labels, type='C-classification', kernel = "polynomial", cost = 10, coef0=1, 
               scale = FALSE, probability = T ) 
svmfit2 <- svm(labels ~ ., data = trainset, kernel = "radial", cost = 100, gamma=1e-08,  type='C-classification', scale = FALSE) 
svmfit3 <- svm(labels ~ ., data = trainset, kernel = "sigmoid", cost = 10, scale = FALSE, coef0=1, type='C-classification') 

plot(svmfit)
ct <- table(tt$labels, predict(svmfit2))  # tabulate
precision <- ct[4]/sum(ct[3:4])
recall <- ct[4]/sum(ct[c(2,4)])
(F <- 2*precision*recall/(precision+recall)) 

### Tuning
set.seed(100) 
tuned <- tune.svm(labels ~., data = tt, gamma = 10^(-10:-7), cost = 10^(2:3)) # tune
summary (tuned) # to select best gamma and cost
plot(tuned)

####################################################
################# MODEL FOUR #######################
################ Neural Network ####################
####################################################

############### usage ##############################
## S3 method for class 'formula'
nnet(formula, data, weights, ...,
     subset, na.action, contrasts = NULL)
## Default S3 method:
nnet(x, y, weights, size, Wts, mask,
     linout = FALSE, entropy = FALSE, softmax = FALSE,
     censored = FALSE, skip = FALSE, rang = 0.7, decay = 0,
     maxit = 100, Hess = FALSE, trace = TRUE, MaxNWts = 1000,
     abstol = 1.0e-4, reltol = 1.0e-8, ...)
############### my model ###########################
nn1 <- nnet(labels~.,trainset, weights, size=10, Wts, linout = F, entropy = F,
     softmax = F, skip = F, rang = 0.7,decay = 0.015, maxit = 300, MaxNWts=5000,
     trace = T)
nn2<-nnet(labels~.,tt,size=15,decay=0.01,maxit=500,linout=T,trace=F,MaxNWts=8000)
train_preds <- ifelse(nn1$fitted.values>0.5, 1, 0)
table(trainset$labels, train_preds)

preds <- predict(nn1, test)
table(ts$labels, predict(nn1, test[-length(test)], type='class'))
ct <- table(tt$labels, predict(svmfit2))  # tabulate
precision <- ct[4]/sum(ct[3:4])
recall <- ct[4]/sum(ct[c(2,4)])
(F <- 2*precision*recall/(precision+recall)) 

####################################################
############### MODEL FIVE ########################
################ XgBoost ###########################
#################################################### 
xgb.train(params = list(booster=gbtree, silent=0, eta=0.3, gamma=), data, nrounds, watchlist = list(), obj = NULL,
          feval = NULL, verbose = 1, print_every_n = 1L,
          early_stopping_rounds = NULL, maximize = NULL, save_period = NULL,
          save_name = "xgboost.model", xgb_model = NULL, callbacks = list(), ...)

xgboost(data = NULL, label = NULL, missing = NA, weight = NULL,
        params = list(), nrounds, verbose = 1, print_every_n = 1L,
        early_stopping_rounds = NULL, maximize = NULL, save_period = 0,
        save_name = "xgboost.model", xgb_model = NULL, callbacks = list(), ...)

xgb1 <- xgboost(data = data.matrix(trainset[ ,-ncol(trainset)]), # input dataset. xgb.train takes only an xgb.DMatrix as the input. xgboost, in addition, also accepts matrix, dgCMatrix, or local data file.
                label = trainset$labels, 
                eta = 0.1,
                max_depth = 15, 
                nround=100, # the max number of iterations
                subsample = 0.3, # add stochatic, more robust,  prevent overfitting
                colsample_bytree = 0.5,  # subsample ratio of columns when constructing each tree. Default: 1
                seed = 1,
                eval_metric = "merror", # evaluation func
                objective = "multi:softprob", # objective func
                base_score = 0.5, # the initial prediction score of all instances, global bias. Default: 0.5
                num_class = 2,
                nthread = 4)

pred <- predict(xgb1, data.matrix(test_model[-length(test_model)]))
# reshape it to a num_class-columns matrix
pred <- matrix(pred, ncol=2, byrow=TRUE)
# convert the probabilities to softmax labels
pred_labels <- max.col(pred) - 1
ct <- table(test_model$labels, pred_labels)
precision <- ct[4]/sum(ct[3:4])
recall <- ct[4]/sum(ct[c(2,4)])
(F <- 2*precision*recall/(precision+recall)) # testset F_measure=0.81

prob <- pred[ ,2]
test_id %>%
  mutate(prob) %>%
  filter(prob>0.3) %>%
  arrange(desc(prob)) -> predict_alllist
# choose the best prob user-sku pairs
test_id %>%
  mutate(prob) %>%
  filter(prob>0.2) %>%
  arrange(desc(prob)) %>%
  group_by(user_id) %>%
  dplyr::slice(1) %>% # chose the highest prob row for each user_id
  arrange(desc(prob)) -> predict_list 
write_csv(predict_alllist, 'Downloads/JData/xgb_all.csv') # 3cols
write_csv(predict_list, 'Downloads/JData/xgb1.csv') # 3cols

## An xgb.train example where custom objective and evaluation metric are used:
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(metric = "error", value = err))
}

## An xgb.train example of using variable learning rates at each iteration:
param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2)
my_etas <- list(eta = c(0.5, 0.1))

# Inspect the prediction error vs number of trees:
lb <- test$label
dtest <- xgb.DMatrix(test$data, label=lb)
err <- sapply(1:25, function(n) {
  pred <- predict(bst, dtest, ntreelimit=n)
  sum((pred > 0.5) != lb)/length(lb)
})
plot(err, type='l', ylim=c(0,0.1), xlab='#trees')





