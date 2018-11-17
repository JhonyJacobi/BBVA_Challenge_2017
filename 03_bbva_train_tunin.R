library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores() -1)


library(doParallel)
registerDoParallel(cores = 3)


trainDF = df_entrenamiento
valDF = df_validación

trainDF$ATTRITION = as.factor(trainDF$ATTRITION)
valDF$ATTRITION = as.factor(valDF$ATTRITION)

levels(trainDF$ATTRITION) = c("usa","nousa")

LogLosSummary = function (data, lev = NULL, model = NULL) {
  LogLos = function(actual, pred, eps = 1e-15) {
    stopifnot(all(dim(actual) == dim(pred)))
    pred[pred < eps] = eps
    pred[pred > 1 - eps] = 1 - eps
    -sum(actual * log(pred)) / nrow(pred) 
  }
  if (is.character(data$obs)) data$obs = factor(data$obs, levels = lev)
  pred = data[, "pred"]
  obs = data[, "obs"]
  isNA = is.na(pred)
  pred = pred[!isNA]
  obs = obs[!isNA]
  data = data[!isNA, ]
  cls = levels(obs)
  
  if (length(obs) + length(pred) == 0) {
    out = rep(NA, 2)
  } else {
    pred = factor(pred, levels = levels(obs))
    require("e1071")
    out = unlist(e1071::classAgreement(table(obs, pred)))[c("diag",                                                                                                                                                             "kappa")]
    
    probs = data[, cls]
    actual = model.matrix(~ obs - 1)
    out2 = LogLos(actual = actual, pred = probs)
  }
  out = c(out, out2)
  names(out) = c("Accuracy", "Kappa", "LogLoss")
  
  if (any(is.nan(out))) out[is.nan(out)] = NA 
  
  out
}

ControlParamteres = trainControl(method = "cv",
                                 number = 5,
                                 savePredictions = TRUE,
                                 classProbs = TRUE,
                                 summaryFunction = LogLosSummary
                                 )

parametersGrid =  expand.grid(eta = c(0.05,0.06),
                               colsample_bytree=c(0.5,0.7),
                               max_depth=c(6,9),
                               nrounds=100,
                               gamma=1,
                               min_child_weight=10,
                               subsample = 1
                               )

set.seed(123)
modelxgboost = train(ATTRITION~., 
                     data = trainDF,
                     method = "xgbTree",
                     objective = "binary:logistic",
                     #eval_metric = "logloss",
                     metric = "LogLoss",
                     maximize = FALSE,
                     trControl = ControlParamteres,
                     tuneGrid=parametersGrid,
                     na.action = na.pass)

modelxgboost


modellog = train(ATTRITION~.,
                 data=trainDF,
                 method="glm",
                 family=binomial(link='logit'),
                 trControl= ControlParamteres,
                 na.action = na.pass)

predictions = predict(modelxgboost,trainDF,type = "prob")


#predictions = apply(predictions, c(1,2), function(x) min(max(x, 1E-15), 1-1E-15))

logLoss(as.numeric(valDF$ATTRITION), predictions[,1])


results = resamples(list(XGboost=modelxgboost, LogisticRegression=modellog))
bwplot(results, metric="LogLoss")
bwplot(results, metric="Accuracy")

# Logistic regression
preds.log = predict.train(modellog, newdata=valDF, type="prob") 

## CV

bst_cv = xgb.cv (params = params, 
                 data = dtrain,
                 nfold = 5,
                 nrounds = 100,
                 #watchlist = list(eval = dvalidation, train = dtrain),
                 maximize = F,
                 missing = NA
                 )


