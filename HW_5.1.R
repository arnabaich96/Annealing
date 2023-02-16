library(readr)
library(dplyr)
library(stargazer)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)
library(job)
library(dplyr)
library(torch)
library(tensorA)
library(doParallel)

train_X <-
  read_table(
    "D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/Gisette/gisette_train.data",
    col_names = FALSE
  )

test_X <-
  read_table(
    "D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/Gisette/gisette_valid.data",
    col_names = FALSE
  )
train_Y <-
  read_csv(
    "D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/Gisette/gisette_train.labels",
    col_names = FALSE
  )

test_Y <-
  read_table(
    "D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/Gisette/gisette_valid.labels",
    col_names = FALSE
  )

# column means and Sds
x_mean=as.numeric(colMeans(train_X[,-5001]))
x_sd =as.numeric(apply(train_X[,-5001],2,sd))

b=scale(train_X[, -5001],center=x_mean,scale=x_sd)
X = rbind(scale(train_X[, -5001],center=x_mean,scale=x_sd),
          scale(test_X[, -5001],center=x_mean,scale=x_sd))
X = X[, colSums(is.na(X)) == 0]
# setting up train data
X_train = X[1:6000, ]
X_train = cbind(X0 = rep(1, nrow(X_train)), X_train)
# dim(X_train)
data_train = list(y = as.matrix(train_Y), x = as.matrix(X_train))
# str(data_train)
#setting up test data
X_test = X[6001:7000, ]
X_test =  cbind(X0 = rep(1, nrow(X_test)), X_test)
# dim(X_test)
data_test = list(y = as.matrix(test_Y), x = as.matrix(X_test))
# str(data_test)

n_iter=300
eta=1/nrow(data_train$y)
mu=100
k=500
w_init = as.matrix(rep(0, ncol(data_train$x)))



annealing = function(n_iter=300,k=30,data_train,data_test,
                    eta=1/nrow(data_train$y),mu=100,
                    w_init = as.matrix(rep(0, ncol(data_train$x))))
{
  w_new = w_init
  n = 1
  x_tr_new = data_train$x
  x_te_new  = data_test$x
  my.list = list()
  M = array()
  train_miss = array()
  iteration = array()
  y_train = data_train$y
  y_test = data_test$y
  
  # Updating w
  
  while (n <= n_iter)
  {
    x_tr = as.matrix(x_tr_new)
    x_te = as.matrix(x_te_new)
    w = w_new
    pred = x_tr %*% w
    #Calculating Gradient
    v = as.matrix(y_train - (1+exp(-pred))**(-1))
    grad = t(x_tr) %*% v
    #updating coefficients
    w_new_theta = w + as.matrix(eta*grad)
    w_new_sign = sign(w_new_theta)
    M[n] = k + (nrow(w_new)-k)*max(0,(n_iter-2*n)/(2*n*mu + n_iter))
    df1 = data.frame(cbind(w_new_theta**2,w_new_sign ,t(x_tr)))
    df2 = data.frame(cbind(w_new_theta**2,w_new_sign ,t(x_te)))
    d1= top_n(df1,M[n],X1)
    d2= top_n(df2,M[n],X1) 
    x_tr_new = t(d1[,-c(1,2)])
    x_te_new = t(d2[,-c(1,2)])
    w_new = as.matrix(sqrt(d1[,1])*d1[,2])
 # updating  iteration
   n = n + 1
  } 

  ## Train Data
  link_train = x_tr_new %*% w_new
  roc_train = roc(as.numeric(y_train), as.numeric(link_train))
  threshold_train = as.numeric(coords(roc_train, "best", ret = "threshold",drop=TRUE)[1])
  y_hat_train = as.factor(ifelse(link_train > threshold_train, 1, -1))
  levels(y_hat_train) = c("-1", "1")
  cm_train = confusionMatrix(as.factor(y_train), as.factor(y_hat_train))
  my.list$train_miss = as.numeric(1 - cm_train$byClass['Balanced Accuracy'])
  ## Test Data
  link_test = x_te_new %*% w_new
  # prob_test = exp(link_test)/(1+exp(link_test))
  roc_test = roc(as.numeric(y_test), as.numeric(link_test))
  threshold_test = as.numeric(coords(roc_test, "best", ret = "threshold",drop=TRUE)[1])
  y_hat_test = as.factor(ifelse(link_test > threshold_test, 1, -1))
  levels(y_hat_test) = c("-1", "1")
  cm_test = confusionMatrix(as.factor(y_test), as.factor(y_hat_test))
  my.list$test_miss = as.numeric(1 - cm_test$byClass['Balanced Accuracy'])
  return(my.list)
}

# annealing(n_iter=300,k=500,
#           data_train,data_test,
#           eta=1/nrow(data_train$y),mu=100,
#           w_init = as.matrix(rep(0, ncol(data_train$x))))

k_all = c(10,30,100,300,500)
final_annealing = function(k1,n_iter1=300,eta1=1/nrow(data_train$y),mu1=100,
                              w_init1 = as.matrix(rep(0, ncol(data_train$x))))
{
  w_init = as.matrix(rep(0, ncol(data_train$x)))
  r1 = annealing(k=k1,n_iter1, data_train, data_test,eta = eta1,mu= mu1,w_init = w_init1)
  
  table = data.frame(k1,r1$train_miss,r1$test_miss)
  colnames(table) = c("Feature","Miss.Train","Miss.Test")
  
  return(table)
}


my.cluster <- makeCluster(10)
registerDoParallel(my.cluster)
clusterExport(my.cluster,c("final_annealing","annealing","data_train","data_test"),envir = .GlobalEnv)
invisible(clusterEvalQ(my.cluster,
                       {library(dplyr)
                         library(stargazer)
                         library(caret)
                         library(pROC)
                       }))
u=parSapply(my.cluster,k_all,final_annealing,n_iter1=300,eta1=1/nrow(data_train$y)
            ,mu1=100, w_init1 = as.matrix(rep(0, ncol(data_train$x))))

D=data.frame(t(u))
D
stopCluster(my.cluster)
#train vs test missplot
plot = ggplot(D,aes(x= as.numeric(unlist(D['Feature']))))+
  geom_line(aes(y = as.numeric(unlist(D['Miss.Train'])),color = "Training"))+
  geom_line(aes(y = as.numeric(unlist(D['Miss.Test'])),color = "Testing"))+
  ylim(0,0.3) + ylab('Misclassification Error') + xlab('Number of Feature')
plot

data=data_train
# training Loss Plot
Loss_plot = function(n_iter=300,data,k=300,s=0.001,eta=0.001,mu=100)
{
  w_new = as.matrix(rep(0, ncol(data$x)))
  n = 1
  x_new = data$x
  iteration = array()
  y = data$y
  loss = array()
  M = array()
  # Updating w
  
  while (n <= n_iter)
  {
    x = x_new
    w = w_new
    link = x %*% w
    #Calculating Gradient
    v = as.matrix(y- (1+exp(-link))^(-1))
    grad = t(x) %*% v
    #updating coefficients
    w_new_theta = w - (eta*grad)
    w_new_sign = sign(w_new_theta)
    M[n] = k + (nrow(w_new)-k)*max(0,(n_iter-2*n)/(2*n*mu + n_iter))
    df = data.frame(cbind(w_new_theta^2,w_new_sign,t(x)))
    d = top_n(df,M[n],X1)
    x_new = t(d[,-c(1,2)])
    w_new = as.matrix(sqrt(d[,1])*d[,2])
    loss[n] = (nrow(y)**(-1))*(log(1+exp(t(y) %*% x_new %*% w_new))- t(y) %*% x_new %*% w_new)
                +s*sqrt(sum(w_new**2))
    iteration[n]=n
    n = n + 1
  } 
  out=data.frame(iteration,loss)
  plot1 = ggplot(out, aes(x = iteration)) + geom_line(aes(y = loss)) + 
           ylab('Loss') + xlab('Iteration')+ ylim(min(loss)-1,max(loss)+1)

  return(plot1)
}
Loss_plot(k=300,data=data_train)

# Roc Plot
Roc_plot = function(n_iter=300,eta=1/nrow(data_train$y),k=300,train,test
                    ,mu=100, w_init = as.matrix(rep(0, ncol(data_train$x))))
{
  
  w_new = w_init
  n = 1
  x_tr_new = train$x
  x_te_new  = test$x
  M = array()
  train_miss = array()
  iteration = array()
  y_train =train$y
  y_test = test$y
  
  # Updating w
  
  while (n <= n_iter)
  {
    x_tr = as.matrix(x_tr_new)
    x_te = as.matrix(x_te_new)
    w = w_new
    pred = x_tr %*% w
    v = as.matrix(y_train- (1+exp(-pred))^(-1))
    grad = t(x_tr) %*% v
    w_new_theta = w - as.matrix(eta*grad)
    w_new_sign = sign(w_new_theta)
    M[n] = k + (nrow(w_new)-k)*max(0,(n_iter-2*n)/(2*n*mu + n_iter))
    df1 = data.frame(cbind(w_new_theta**2,w_new_sign ,t(x_tr)))
    df2 = data.frame(cbind(w_new_theta**2,w_new_sign ,t(x_te)))
    d1= top_n(df1,M[n],X1)
    d2= top_n(df2,M[n],X1) 
    x_tr_new = t(d1[,-c(1,2)])
    x_te_new = t(d2[,-c(1,2)])
    w_new = as.matrix(sqrt(d1[,1])*d1[,2])
    n = n + 1
  } 
  link_train = x_tr_new %*% w_new
  roc_train = roc(as.numeric(y_train), as.numeric(link_train))
  link_test = x_te_new%*% w_new
  roc_test = roc(as.numeric(y_test), as.numeric(link_test))
  plot = ggroc(list(Train = roc_train, Test = roc_test ))+
         geom_abline(slope=1,intercept = 1,color = "blue")+
         ggtitle('For 300 Features')
  return(plot)
}

Roc_plot(n_iter=300,eta=1/nrow(data_train$y),k=300,train=data_train
         ,test=data_test,m=100, w_init= as.matrix(rep(0, ncol(data_train$x))))




