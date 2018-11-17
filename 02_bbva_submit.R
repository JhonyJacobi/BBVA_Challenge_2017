rm(list = ls())
gc()

setwd("D:/Competencia/BBVA")
getwd()
library(data.table)
library(dplyr)
library(dummies)
library(lubridate)
library(caret)
library(xgboost)

# 1. CARGAR DATA
df_test_cli_aux = fread("01_Data/test_clientes.csv")
df_test_req_aux = fread("01_Data/test_requerimientos.csv")

# 2. DEFINICIÓN Y CREACIÓN DE VARIABLES
df_test_cli_aux$FLAG_LIMA_PROVINCIA[df_test_cli_aux$FLAG_LIMA_PROVINCIA == ""] = "NA"
df_test_cli_aux$RANG_INGRESO[df_test_cli_aux$RANG_INGRESO == ""] = "NA"

# 2.1 Generar dummies
dim(df_test_cli_aux)
df_test_cli_aux_dumm = dummy.data.frame(df_test_cli_aux, sep = "_")

# 2.2 Generar variables transaccionales
df_test_req_aux_uniq = 
  df_test_req_aux %>%
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
df_test_cli_aux_dumm_req = left_join(df_test_cli_aux_dumm, df_test_req_aux_uniq, by = "ID_CORRELATIVO")

# Definir campos de tipo fecha
df_test_cli_aux_dumm_req = mutate(df_test_cli_aux_dumm_req, 
                                  CODMES_DATE = ymd(paste(as.character(df_test_cli_aux_dumm_req$CODMES),"01",sep = ""))
                                  )
df_test_cli_aux_dumm_req = mutate(df_test_cli_aux_dumm_req, 
                                  REQ_ULT_MES_DATE = ymd(paste(as.character(df_test_cli_aux_dumm_req$REQ_ULT_MES),"01",sep = ""))
                                  )
df_test_cli_aux_dumm_req = mutate(df_test_cli_aux_dumm_req, 
                                  REQ_INI_MES_DATE = ymd(paste(as.character(df_test_cli_aux_dumm_req$REQ_INI_MES),"01",sep = ""))
                                  )

# Generar variables derivadas de fechas
df_test_cli_aux_dumm_req = mutate(df_test_cli_aux_dumm_req, 
                                  DIAS_ULT_REQ = as.numeric(CODMES_DATE - REQ_ULT_MES_DATE)
                                  )
df_test_cli_aux_dumm_req = mutate(df_test_cli_aux_dumm_req, 
                                  DIAS_INI_REQ = as.numeric(CODMES_DATE - REQ_INI_MES_DATE)
                                  )

# Generar variables de saldo
df_test_cli_aux_dumm_req$SDO_OTSSFF_NRO_MESES = apply(df_test_cli_aux_dumm_req[,77:82], 1, sum)

# Generar variables de accesos por canal
df_test_cli_aux_dumm_req$NRO_ACCES_CANAL1_SUM = apply( df_test_cli_aux_dumm_req[,53:58], 1, sum)
df_test_cli_aux_dumm_req$NRO_ACCES_CANAL1_SD = apply( df_test_cli_aux_dumm_req[,53:58], 1, sd)

df_test_cli_aux_dumm_req$NRO_ACCES_CANAL2_SUM = apply( df_test_cli_aux_dumm_req[,59:64], 1, sum)
df_test_cli_aux_dumm_req$NRO_ACCES_CANAL2_SD = apply( df_test_cli_aux_dumm_req[,59:64], 1, sd)

df_test_cli_aux_dumm_req$NRO_ACCES_CANAL3_SUM = apply( df_test_cli_aux_dumm_req[,65:70], 1, sum)
df_test_cli_aux_dumm_req$NRO_ACCES_CANAL3_SD = apply( df_test_cli_aux_dumm_req[,65:70], 1, sd)

df_test_cli_aux_dumm_req$NRO_ACCES_CANALES = apply( df_test_cli_aux_dumm_req[,c(53:58,59:64,65:70)], 1, sum)

df_test_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS0_SUM = apply(df_test_cli_aux_dumm_req[,c(53,59,65)], 1, sum)
df_test_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS1_SUM = apply(df_test_cli_aux_dumm_req[,c(54,60,66)], 1, sum)
df_test_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS2_SUM = apply(df_test_cli_aux_dumm_req[,c(55,61,67)], 1, sum)
df_test_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS3_SUM = apply(df_test_cli_aux_dumm_req[,c(56,62,68)], 1, sum)
df_test_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS4_SUM = apply(df_test_cli_aux_dumm_req[,c(57,63,69)], 1, sum)
df_test_cli_aux_dumm_req$NRO_ACCES_CANALES_MENOS5_SUM = apply(df_test_cli_aux_dumm_req[,c(58,64,70)], 1, sum)

# Generar variable de nro de entidades financieras
df_test_cli_aux_dumm_req$NRO_ENTID_SSFF_AVG = apply(df_test_cli_aux_dumm_req[,71:76], 1, mean)
df_test_cli_aux_dumm_req$NRO_ENTID_SSFF_SD = apply(df_test_cli_aux_dumm_req[,71:76], 1, sd)

# Generar variable transformada de EDAD
df_test_cli_aux_dumm_req$EDAD_LN = round(log(df_test_cli_aux_dumm_req$EDAD),1)
df_test_cli_aux_dumm_req$ANTIGUEDAD_LN = round(log(df_test_cli_aux_dumm_req$ANTIGUEDAD),1)

# Generar variable de seguros
df_train_aux$SEGURO_NRO_MESES = apply(df_train_aux[,39:44], 1, sum)

# 2.3 Tratamiento de missing
df_test_cli_aux_dumm_req$REQ_NUM_PROD[is.na( df_test_cli_aux_dumm_req$REQ_NUM_PROD)] = 0
df_test_cli_aux_dumm_req$REQ_NUM_SUBMOT[is.na( df_test_cli_aux_dumm_req$REQ_NUM_SUBMOT)] = 0
df_test_cli_aux_dumm_req$REQ_NUM_REC[is.na( df_test_cli_aux_dumm_req$REQ_NUM_REC)] = 0
df_test_cli_aux_dumm_req$REQ_NUM_SOL[is.na( df_test_cli_aux_dumm_req$REQ_NUM_SOL)] = 0
df_test_cli_aux_dumm_req$REQ_NUM_PROC_TOT[is.na( df_test_cli_aux_dumm_req$REQ_NUM_PROC_TOT)] = 0
df_test_cli_aux_dumm_req$REQ_NUM_PROC_PAR[is.na( df_test_cli_aux_dumm_req$REQ_NUM_PROC_PAR)] = 0
df_test_cli_aux_dumm_req$REQ_NUM_PROC_NOP[is.na( df_test_cli_aux_dumm_req$REQ_NUM_PROC_NOP)] = 0
df_test_cli_aux_dumm_req$REQ_NUM_TOT[is.na( df_test_cli_aux_dumm_req$REQ_NUM_TOT)] = 0

# 2.4 Selección de variables por definición
df_test_aux = dplyr::select(df_test_cli_aux_dumm_req, 
                            -ID_CORRELATIVO,
                            -CODMES,
                            -REQ_ULT_MES,
                            -REQ_INI_MES,
                            -CODMES_DATE,
                            -REQ_ULT_MES_DATE,
                            -REQ_INI_MES_DATE
                            )

# Generar variable de cantidad de NA´s
df_test_aux = mutate(df_test_aux, NRO_NA = rowSums(is.na(df_test_aux)))

# Generar variable de interaccion Atiguedad y Edad
df_test_aux = mutate(df_test_aux, ANTI_EDAD = ANTIGUEDAD*EDAD)

# Generar variable de tendencias
slope  =  function(x){
  if(all(is.na(x)))
    # if x is all missing, then lm will throw an error that we want to avoid
    return(NA)
  else
    return(coef(lm(I(1:6)~x))[2])
}

df_test_aux$NRO_ENTID_SSFF_SLOPE = apply(df_test_aux[,c("NRO_ENTID_SSFF_MENOS0","NRO_ENTID_SSFF_MENOS1","NRO_ENTID_SSFF_MENOS2",
                                                        "NRO_ENTID_SSFF_MENOS3","NRO_ENTID_SSFF_MENOS4","NRO_ENTID_SSFF_MENOS5")],
                                         1,slope)
df_test_aux$SDO_ACTIVO_SLOPE = apply(df_test_aux[,c("SDO_ACTIVO_MENOS0","SDO_ACTIVO_MENOS1","SDO_ACTIVO_MENOS2",
                                                    "SDO_ACTIVO_MENOS3","SDO_ACTIVO_MENOS4","SDO_ACTIVO_MENOS5")],
                                     1,slope)
df_test_aux$NRO_ACCES_CANAL1_SLOPE = apply(df_test_aux[,c("NRO_ACCES_CANAL1_MENOS0","NRO_ACCES_CANAL1_MENOS1","NRO_ACCES_CANAL1_MENOS2",
                                                          "NRO_ACCES_CANAL1_MENOS3","NRO_ACCES_CANAL1_MENOS4","NRO_ACCES_CANAL1_MENOS5")],
                                           1,slope)
df_test_aux$NRO_ACCES_CANAL2_SLOPE = apply(df_test_aux[,c("NRO_ACCES_CANAL2_MENOS0","NRO_ACCES_CANAL2_MENOS1","NRO_ACCES_CANAL2_MENOS2",
                                                          "NRO_ACCES_CANAL2_MENOS3","NRO_ACCES_CANAL2_MENOS4","NRO_ACCES_CANAL2_MENOS5")],
                                           1,slope)
df_test_aux$NRO_ACCES_CANAL3_SLOPE = apply(df_test_aux[,c("NRO_ACCES_CANAL3_MENOS0","NRO_ACCES_CANAL3_MENOS1","NRO_ACCES_CANAL3_MENOS2",
                                                          "NRO_ACCES_CANAL3_MENOS3","NRO_ACCES_CANAL3_MENOS4","NRO_ACCES_CANAL3_MENOS5")],
                                           1,slope)
df_test_aux$NRO_ACCES_CANALES_SLOPE = apply(df_test_aux[,c("NRO_ACCES_CANALES_MENOS0_SUM","NRO_ACCES_CANALES_MENOS1_SUM","NRO_ACCES_CANALES_MENOS2_SUM",
                                                           "NRO_ACCES_CANALES_MENOS3_SUM","NRO_ACCES_CANALES_MENOS4_SUM","NRO_ACCES_CANALES_MENOS5_SUM")],
                                            1,slope)

df_test_aux$NRO_ENTID_SSFF_SLOPE[is.na(df_test_aux$NRO_ENTID_SSFF_SLOPE)] = 0
df_test_aux$SDO_ACTIVO_SLOPE[is.na(df_test_aux$SDO_ACTIVO_SLOPE)] = 0
df_test_aux$NRO_ACCES_CANAL1_SLOPE[is.na(df_test_aux$NRO_ACCES_CANAL1_SLOPE)] = 0
df_test_aux$NRO_ACCES_CANAL2_SLOPE[is.na(df_test_aux$NRO_ACCES_CANAL2_SLOPE)] = 0
df_test_aux$NRO_ACCES_CANAL3_SLOPE[is.na(df_test_aux$NRO_ACCES_CANAL3_SLOPE)] = 0
df_test_aux$NRO_ACCES_CANALES_SLOPE[is.na(df_test_aux$NRO_ACCES_CANALES_SLOPE)] = 0

# 3. REPLICA MODELO XGBOOST

pred_test_bst = predict(bst, as.matrix(df_test_aux[features]))

# 5. Inputación EDAD y ANTIGUEDAD

EDAD_pred_test = round(predict(EDAD_mod, df_test_cli_aux_dumm[is.na(df_test_cli_aux_dumm$EDAD), ]),0)
ANTIGUEDAD_pred_test = predict(ANTIGUEDAD_mod, df_test_cli_aux_dumm[is.na(df_test_cli_aux_dumm$ANTIGUEDAD), ])

df_test_cli_aux1 = left_join(data.frame(cbind(df_test_cli_aux_dumm,ID = row.names(df_test_cli_aux_dumm)), stringsAsFactors = FALSE), 
                                  data.frame(EDAD_pred_test, ID = as.character(names(EDAD_pred_test)), stringsAsFactors = FALSE), 
                                  by = "ID",
                                  all.x = TRUE)

df_test_cli_aux1 = left_join(df_test_cli_aux1, 
                                  data.frame(ANTIGUEDAD_pred_test, ID = as.character(names(ANTIGUEDAD_pred_test)), stringsAsFactors = FALSE), 
                                  by = "ID",
                                  all.x = TRUE)

df_test_cli_aux1$EDAD_NEW = round(apply(df_test_cli_aux1[,c(17,84)], 1, sum, na.rm = T),0)
df_test_cli_aux1$ANTIGUEDAD_NEW = round(apply(df_test_cli_aux1[,c(18,85)], 1, sum, na.rm = T),0)

# 6. REPLICA MODELO LOGISTICO y RANDOM FOREST

pred_test_log = predict(model_log, newdata = df_test_cli_aux1[feature_log], type='response')
pred_test_rf = predict(model_rf, newdata = df_test_cli_aux1[feature_log], type='prob')[,2]

# 7. GUARDAR RESULTADOS FINALES
df_stacking_test = data.frame(pred_rf = pred_test_rf, pred_test_log, pred_test_bst)

df_submit = data.frame(ID_CORRELATIVO = df_test_cli_aux1$ID_CORRELATIVO,
                       #ATTRITION = apply(df_stacking_test, 1, mean))
                       ATTRITION = pred_test_bst)

write.csv(df_submit, file = "03_Resultado/bbva_sumbit_07.csv", quote = FALSE, row.names = FALSE)
