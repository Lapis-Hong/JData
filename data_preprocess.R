library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(nnet)
library(randomForest)
library(xgboost)
library(e1071)

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

#####################################################
######## data cleaning ##############################
#####################################################

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
  # turn age, sex into dummy variable
  age_vector <- tbl_df(class.ind(tbl_cate8$age))
  sex_vector <- tbl_df(class.ind(tbl_cate8$sex))
  colnames(sex_vector) <- c('M', 'F', 'unknown')
  tbl_dummy <- select(bind_cols(tbl_cate8, age_vector, sex_vector), -age, -sex)
  # turn date into days 
  temp <- mutate(tbl_dummy, user_reg_tm=as.integer(as.Date('2016-04-15')-user_reg_tm))
  # delete the useless user and sku
  tbl_clean <- inner_join(inner_join(temp, sku_keep), user_keep)
  return(tbl_clean)
}
tbl_clean <- data_clear(tbl_cate8)
# tbl_clean <- read_csv('Downloads/JData/tbl_clean.csv', locale=locale(tz=Sys.timezone())) 

####################################################
########## data anlysis ############################
####################################################

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

######################################################
######## feature extraction ##########################
######################################################
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
  filter(data, time<time[length(time)]-24*60*60, time>time[length(time)]-3*24*60*60)
}
latest_7 <- function(data){
  filter(data, time<time[length(time)]-3*24*60*60, time>time[length(time)]-7*24*60*60)
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
           r1=`4`/(`1`+`6`), r2=`2`/(`1`+`6`), 
           r3=`5`/(`1`+`6`), r4=`4`/(`2`+`5`)) -> data 
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


##########################################################
######## Data Preprocessing  #############################
##########################################################

###### replacing na with zero
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

##################################################################
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
##################################################################
normalization <- function(data){
  center <- sweep(data, 2, apply(data, 2, min),'-') #在列的方向上减去最小值，不加‘-’也行
  R <- apply(data, 2, max) - apply(data, 2, min) #算出极差，即列上的最大值-最小值
  data_nor<- sweep(center, 2, R, "/") #把减去均值后的矩阵在列的方向上除以极差向量
  return(data_nor)
}
trainset_nor <- normalization(replace_na(trainset_1, list(0))) 

# create labels
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

# create trainset and testset
create_trainset <- function(from='2016-02-01' ,last=7){
  from <- as.POSIXct(from)
  train <- filter(tbl_clean, time>from & time<(from+last*24*60*60))
  label <- filter(tbl_clean, time>(from+last*24*60*60) & time<(from+(last+5)*24*60*60))
  train <- distinct(trainset_with_feature(train)[-c(3:5)])
  brand_vec <- class.ind(train$brand)
  train_f <- bind_cols(train, tbl_df(brand_vec))
  trainset <- left_join(train_f, create_label(label))[-c(1,2,3)] # del user_id, sku_id, brand
  trainset_scale <- tbl_df(replace(scale(trainset), is.na(scale(trainset)), 0))
  trainset_scale <- mutate(trainset_scale, labels=ifelse(labels>0, 1, 0))
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

test <- filter(tbl_clean, time>'2016-04-08' & time<'2016-04-16')
test <- distinct(trainset_with_feature(test)[-c(3:5)])
brand_vec <- class.ind(test$brand)
test <- bind_cols(test, tbl_df(brand_vec))[-3]
test <- replace(scale(test), is.na(scale(test)), 0)
write_csv(test, 'Downloads/JData/test.csv')


############## imbalance ##########################
# ratio of labels ==1 
ratio <- function(data){
  sum(data$labels==1)/nrow(data)
}

# train_oversampling
train_oversampling <- function(data){
  data %>%
    filter(labels==0) %>%
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
# row bind match by name, produce NA
train_all <- bind_rows(train9, train10, train8, train7, train6, 
                       train5, train4, train3, train2,train1)[-length(train_all)]
train_all[is.na(train_all)] <- 0
write_csv(train_all, 'Downloads/JData/train_all.csv')


# divide training and testing set
set.seed(12345)  
sub <- sample(1:nrow(train_all),round(nrow(train_all)/4))  
trainset <- train_all[-sub, ]
testset <- train_all[sub, ]
