rm(list = ls())
gc()

setwd("D:/Competencia/BBVA")
getwd()

library(data.table)
library(dplyr)
library(dummies)
library(lubridate)
library(caret)
library(MLmetrics)
library(xgboost)
library(randomForest)
#install.packages("MLmetrics")

# 1. CARGAR DATA TRAIN Y TEST
df_train_cli_aux = fread("01_Data/train_clientes.csv")
df_train_req_aux = fread("01_Data/train_requerimientos.csv")

# 2. DEFINICIÓN Y CREACIÓN DE VARIABLES
df_train_cli_aux$FLAG_LIMA_PROVINCIA[df_train_cli_aux$FLAG_LIMA_PROVINCIA == ""] = "NA"
df_train_cli_aux$RANG_INGRESO[df_train_cli_aux$RANG_INGRESO == ""] = "NA"

# 2.1 Generar dummies
dim(df_train_cli_aux)
df_train_cli_aux_dumm = dummy.data.frame(df_train_cli_aux, sep = "_")

# 2.2 Generar variables transaccionales
df_train_req_aux_uniq = 
df_train_req_aux %>%
  group_by(ID_CORRELATIVO) %>%
  dplyr::summarise(REQ_ULT_MES = max(CODMES),
                   REQ_INI_MES= min(CODMES),
                   REQ_NUM_PROD = n_distinct(PRODUCTO_SERVICIO_2),
                   REQ_NUM_SUBMOT = n_distinct(SUBMOTIVO_2),
                   REQ_NUM_REC = sum(TIPO_REQUERIMIENTO2 == "Reclamo"),
                   REQ_NUM_SOL = sum(TIPO_REQUERIMIENTO2 == "Solicitud"),
                   REQ_NUM_PROC_TOT = sum(DICTAMEN == "PROCEDE TOTAL"),
                   REQ_NUM_PROC_PAR = sum(DICTAMEN == "PROCEDE PARCIAL"),
                   REQ_NUM_PROC_NOP = sum(DICTAMEN == "NO PROCEDE"),
                   REQ_NUM_TOT = n()
                   )

# Tabla final con los requerimientos
df_train_cli_aux_dumm_req = left_join(df_train_cli_aux_dumm, df_train_req_aux_uniq, by = "ID_CORRELATIVO")

# Definir campos de tipo fecha
df_train_cli_aux_dumm_req = mutate(df_train_cli_aux_dumm_req, 
                                   CODMES_DATE = ymd(paste(as.character(df_train_cli_aux_dumm_req$CODMES),"01",sep = ""))
                                   )
df_train_cli_aux_dumm_req = mutate(df_train_cli_aux_dumm_req, 
                                   REQ_ULT_MES_DATE = ymd(paste(as.character(df_train_cli_aux_dumm_req$REQ_ULT_MES),"01",sep = ""))
                                   )
df_train_cli_aux_dumm_req = mutate(df_train_cli_aux_dumm_req, 
                                   REQ_INI_MES_DATE = ymd(paste(as.character(df_train_cli_aux_dumm_req$REQ_INI_MES),"01",sep = ""))
                                   )

# Generar variables derivadas de fechas
df_train_cli_aux_dumm_req = mutate(df_train_cli_aux_dumm_req, 
                                   DIAS_ULT_REQ = round(as.numeric(CODMES_DATE - REQ_ULT_MES_DATE)/30,0)
                                   )
df_train_cli_aux_dumm_req = mutate(df_train_cli_aux_dumm_req, 
                                   DIAS_INI_REQ = round(as.numeric(CODMES_DATE - REQ_INI_MES_DATE)/30,0)
                                   )

# Generar variables de saldo
df_train_cli_aux_dumm_req$SDO_OTSSFF_NRO_MESES = apply(df_train_cli_aux_dumm_req[,78:83], 1, sum)

# Generar variables de accesos por canal
df_train_cli_aux_dumm_req$NRO_ACCES_CANAL1_SUM = apply(df_train_cli_aux_dumm_req[,54:59], 1, sum)
df_train_cli_aux_dumm_req$NRO_ACCES_CANAL1_SD = apply(df_train_cli_aux_dumm_req[,54:59], 1, sd)
                                    
df_train_cli_aux_dumm_req$NRO_ACCES_CANAL2_SUM = apply(df_train_cli_aux_dumm_req[,60:65], 1, sum)
df_train_cli_aux_dumm_req$NRO_ACCES_CANAL2_SD = apply(df_train_cli_aux_dumm_req[,60:65], 1, sd)

df_train_cli_aux_dumm_req$NRO_ACCES_CANAL3_SUM = apply(df_train_cli_aux_dumm_req[,66:71], 1, sum)
df_train_cli_aux_dumm_req$NRO_ACCES_CANAL3_SD = apply(df_train_cli_aux_dumm_req[,66:71], 1, sd)

df_train_cli_aux_dumm_req$NRO_ACCES_CANALES = apply(df_train_cli_aux_dumm_req[,c(54:59,60:65,66:71)], 1, sum)

df_train_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS0_SUM = apply(df_train_cli_aux_dumm_req[,c(54,60,66)], 1, sum)
df_train_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS1_SUM = apply(df_train_cli_aux_dumm_req[,c(55,61,67)], 1, sum)
df_train_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS2_SUM = apply(df_train_cli_aux_dumm_req[,c(56,62,68)], 1, sum)
df_train_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS3_SUM = apply(df_train_cli_aux_dumm_req[,c(57,63,69)], 1, sum)
df_train_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS4_SUM = apply(df_train_cli_aux_dumm_req[,c(58,64,70)], 1, sum)
df_train_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS5_SUM = apply(df_train_cli_aux_dumm_req[,c(59,65,71)], 1, sum)

# Generar variable de nro de entidades financieras
df_train_cli_aux_dumm_req$NRO_ENTID_SSFF_AVG = apply(df_train_cli_aux_dumm_req[,72:77], 1, mean)
df_train_cli_aux_dumm_req$NRO_ENTID_SSFF_SD = apply(df_train_cli_aux_dumm_req[,72:77], 1, sd)
df_train_cli_aux_dumm_req = mutate(df_train_cli_aux_dumm_req, NRO_ENTID_SSFF_CV = NRO_ENTID_SSFF_SD/NRO_ENTID_SSFF_AVG)

# Generar variable transformada de EDAD
df_train_cli_aux_dumm_req$EDAD_LN = round(log(df_train_cli_aux_dumm_req$EDAD),1)
df_train_cli_aux_dumm_req$ANTIGUEDAD_LN = round(log(df_train_cli_aux_dumm_req$ANTIGUEDAD),1)

# Generar variable de seguros
df_train_aux$SEGURO_NRO_MESES = apply(df_train_aux[,39:44], 1, sum)

# 2.3 Tratamiento de missing
df_train_cli_aux_dumm_req$REQ_NUM_PROD[is.na(df_train_cli_aux_dumm_req$REQ_NUM_PROD)] = 0
df_train_cli_aux_dumm_req$REQ_NUM_SUBMOT[is.na(df_train_cli_aux_dumm_req$REQ_NUM_SUBMOT)] = 0
df_train_cli_aux_dumm_req$REQ_NUM_REC[is.na(df_train_cli_aux_dumm_req$REQ_NUM_REC)] = 0
df_train_cli_aux_dumm_req$REQ_NUM_SOL[is.na(df_train_cli_aux_dumm_req$REQ_NUM_SOL)] = 0
df_train_cli_aux_dumm_req$REQ_NUM_PROC_TOT[is.na(df_train_cli_aux_dumm_req$REQ_NUM_PROC_TOT)] = 0
df_train_cli_aux_dumm_req$REQ_NUM_PROC_PAR[is.na(df_train_cli_aux_dumm_req$REQ_NUM_PROC_PAR)] = 0
df_train_cli_aux_dumm_req$REQ_NUM_PROC_NOP[is.na(df_train_cli_aux_dumm_req$REQ_NUM_PROC_NOP)] = 0
df_train_cli_aux_dumm_req$REQ_NUM_TOT[is.na(df_train_cli_aux_dumm_req$REQ_NUM_TOT)] = 0

# 2.4 Selección de variables por definición
df_train_aux = dplyr::select(df_train_cli_aux_dumm_req, 
                            -ID_CORRELATIVO,
                            -CODMES,
                            -REQ_ULT_MES,
                            -REQ_INI_MES,
                            -CODMES_DATE,
                            -REQ_ULT_MES_DATE,
                            -REQ_INI_MES_DATE
                            )

# Generar variable de cantidad de NA´s
df_train_aux = mutate(df_train_aux, NRO_NA = rowSums(is.na(df_train_aux)))
dim(df_train_aux)

# Generar variable de interaccion Atiguedad y Edad
df_train_aux = mutate(df_train_aux, ANTI_EDAD = ANTIGUEDAD*EDAD)

# Generar variable de tendencias
slope  =  function(x){
  if(all(is.na(x)))
    # if x is all missing, then lm will throw an error that we want to avoid
    return(NA)
  else
    return(coef(lm(I(1:6)~x))[2])
}

df_train_aux$NRO_ENTID_SSFF_SLOPE = apply(df_train_aux[,c("NRO_ENTID_SSFF_MENOS0","NRO_ENTID_SSFF_MENOS1","NRO_ENTID_SSFF_MENOS2",
                                                          "NRO_ENTID_SSFF_MENOS3","NRO_ENTID_SSFF_MENOS4","NRO_ENTID_SSFF_MENOS5")],
                                          1,slope)
df_train_aux$SDO_ACTIVO_SLOPE = apply(df_train_aux[,c("SDO_ACTIVO_MENOS0","SDO_ACTIVO_MENOS1","SDO_ACTIVO_MENOS2",
                                                      "SDO_ACTIVO_MENOS3","SDO_ACTIVO_MENOS4","SDO_ACTIVO_MENOS5")],
                                      1,slope)
df_train_aux$NRO_ACCES_CANAL1_SLOPE = apply(df_train_aux[,c("NRO_ACCES_CANAL1_MENOS0","NRO_ACCES_CANAL1_MENOS1","NRO_ACCES_CANAL1_MENOS2",
                                                            "NRO_ACCES_CANAL1_MENOS3","NRO_ACCES_CANAL1_MENOS4","NRO_ACCES_CANAL1_MENOS5")],
                                            1,slope)
df_train_aux$NRO_ACCES_CANAL2_SLOPE = apply(df_train_aux[,c("NRO_ACCES_CANAL2_MENOS0","NRO_ACCES_CANAL2_MENOS1","NRO_ACCES_CANAL2_MENOS2",
                                                            "NRO_ACCES_CANAL2_MENOS3","NRO_ACCES_CANAL2_MENOS4","NRO_ACCES_CANAL2_MENOS5")],
                                            1,slope)
df_train_aux$NRO_ACCES_CANAL3_SLOPE = apply(df_train_aux[,c("NRO_ACCES_CANAL3_MENOS0","NRO_ACCES_CANAL3_MENOS1","NRO_ACCES_CANAL3_MENOS2",
                                                            "NRO_ACCES_CANAL3_MENOS3","NRO_ACCES_CANAL3_MENOS4","NRO_ACCES_CANAL3_MENOS5")],
                                            1,slope)
df_train_aux$NRO_ACCES_CANALES_SLOPE = apply(df_train_aux[,c("NRO_ACCES_CANALES_MENOS0_SUM","NRO_ACCES_CANALES_MENOS1_SUM","NRO_ACCES_CANALES_MENOS2_SUM",
                                                            "NRO_ACCES_CANALES_MENOS3_SUM","NRO_ACCES_CANALES_MENOS4_SUM","NRO_ACCES_CANALES_MENOS5_SUM")],
                                            1,slope)

df_train_aux$NRO_ENTID_SSFF_SLOPE[is.na(df_train_aux$NRO_ENTID_SSFF_SLOPE)] = 0
df_train_aux$SDO_ACTIVO_SLOPE[is.na(df_train_aux$SDO_ACTIVO_SLOPE)] = 0
df_train_aux$NRO_ACCES_CANAL1_SLOPE[is.na(df_train_aux$NRO_ACCES_CANAL1_SLOPE)] = 0
df_train_aux$NRO_ACCES_CANAL2_SLOPE[is.na(df_train_aux$NRO_ACCES_CANAL2_SLOPE)] = 0
df_train_aux$NRO_ACCES_CANAL3_SLOPE[is.na(df_train_aux$NRO_ACCES_CANAL3_SLOPE)] = 0
df_train_aux$NRO_ACCES_CANALES_SLOPE[is.na(df_train_aux$NRO_ACCES_CANALES_SLOPE)] = 0

# 2.5 Selección de variables por Random Forest

#df_train_aux[is.na(df_train_aux)] = -999

rf = randomForest( as.factor(ATTRITION)~., data = df_train_aux[,-c(15,16,90,91,108,109,111)], ntree=200, do.trace=T, importance=T )
imp = importance(rf, type=1)

#index_good_features = which( imp >0, arr.ind = TRUE)[,1]

good_features = row.names(subset(imp, imp[,1]>0))
good_features = append(good_features,"ATTRITION")
  
save(good_features, file = "good_features.RData")

# 3. MODELAMIENTO

# 3.1 Dividir en entrenamiento y validación
set.seed(123)
trainIndex = createDataPartition(df_train_aux$ATTRITION, p = 0.8, list=FALSE)

df_entrenamiento = df_train_aux[trainIndex, ] # sólo las variables importantes + target
df_validación = df_train_aux[-trainIndex, ] # sólo las variables importantes + target

dtrain = xgb.DMatrix(data = as.matrix(df_entrenamiento[,-17]), label = df_entrenamiento$ATTRITION)
dvalidation = xgb.DMatrix(data = as.matrix(df_validación[,-17]),label = df_validación$ATTRITION)
#dtrain = xgb.DMatrix(data = as.matrix(df_entrenamiento[filter(df_importancia, TOP <= 80)$Feature]), label = df_entrenamiento$ATTRITION)
#dvalidation = xgb.DMatrix(data = as.matrix(df_validación[filter(df_importancia, TOP <= 80)$Feature]),label = df_validación$ATTRITION)

features = colnames(dtrain)

# 3.2 Ejecutar el modelo Xgboost
params = list(booster = "gbtree", 
              objective = "binary:logistic", 
              eta=0.01,#0.02
              eval_metric = "logloss",
              gamma=3,
              max_depth=16,
              min_child_weight=10, 
              subsample=1, 
              colsample_bytree=0.6
              #scale_pos_weight = 1
              #max_delta_step = 10
              )

#registerDoParallel(cores = 3)
set.seed(123)
bst = xgb.train (params = params, 
                 data = dtrain, 
                 nrounds = 800,#301,
                 nthread = 3,
                 watchlist = list(eval = dvalidation, train = dtrain),
                 maximize = F,
                 missing = NA
                 )

set.seed(123)
bst_cv = xgb.cv (params = params, 
                 data = dtrain, 
                 nrounds = 122,
                 early_stopping_rounds = 4,
                 nfold = 5,
                 watchlist = list(eval = dvalidation, train = dtrain),
                 maximize = F,
                 missing = NA,
                 prediction = TRUE
                 )

pred_bst = predict(bst_cv, as.matrix(df_validación[-17]))

LogLoss(pred_bst, df_validación$ATTRITION)


# 3.3 Importancia de variables
mat = xgb.importance (feature_names = colnames(dtrain),model = bst)
xgb.plot.importance (importance_matrix = mat[1:50]) 
df_importancia = data.frame(mat[,1:2], TOP = 1:nrow(mat))

# 4. GUARDAR RESULTADOS FINALES
save(bst, file = "04_Objetos/xgb_model_2.Rdata")
save(features, file = "04_Objetos/features_2.Rdata")

getwd()
