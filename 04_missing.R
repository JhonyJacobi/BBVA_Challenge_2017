install.packages("MLmetrics")

library(rpart)
#library(Hmisc)
#install.packages("Hmisc")

# Inputación EDAD
EDAD_mod = rpart(EDAD ~ ., 
                   data=df_train_cli_aux_dumm[!is.na(df_train_cli_aux_dumm$EDAD), -c(1,2,18,19)],
                   method="anova",
                   na.action=na.omit)

feature_edad = names(EDAD_mod$ordered)
EDAD_pred = round(predict(EDAD_mod, df_train_cli_aux_dumm[is.na(df_train_cli_aux_dumm$EDAD), ]),0)
#EDAD_val_pred = predict(EDAD_mod, df_validación[is.na(df_validación$EDAD), ])

# Inputación ANTIGUEDAD
ANTIGUEDAD_mod = rpart(ANTIGUEDAD ~ .,
                 data=df_train_cli_aux_dumm[!is.na(df_train_cli_aux_dumm$ANTIGUEDAD), -c(1,2,17,19)],
                 method="anova", 
                 na.action=na.omit)

feature_antiguedad = names(ANTIGUEDAD_mod$ordered)
ANTIGUEDAD_pred = predict(ANTIGUEDAD_mod, df_train_cli_aux_dumm[is.na(df_train_cli_aux_dumm$ANTIGUEDAD), ])


# Tabla final con los requerimientos
df_entrenamiento_aux1 = left_join(data.frame(cbind(df_train_cli_aux_dumm,ID = row.names(df_train_cli_aux_dumm)), stringsAsFactors = FALSE), 
                                  data.frame(EDAD_pred, ID = as.character(names(EDAD_pred)), stringsAsFactors = FALSE), 
                                  by = "ID",
                                  all.x = TRUE)


df_entrenamiento_aux1 = left_join(df_entrenamiento_aux1, 
                                  data.frame(ANTIGUEDAD_pred, ID = as.character(names(ANTIGUEDAD_pred)), stringsAsFactors = FALSE), 
                                  by = "ID",
                                  all.x = TRUE)

df_entrenamiento_aux1$EDAD_NEW = round(apply(df_entrenamiento_aux1[,c(17,85)], 1, sum, na.rm = T),0)
df_entrenamiento_aux1$ANTIGUEDAD_NEW = round(apply(df_entrenamiento_aux1[,c(18,86)], 1, sum, na.rm = T),0)

# Regresion Logistica

df_entrenamiento_rl = df_entrenamiento_aux1[ trainIndex, ] # sólo las variables importantes + target
df_validación_rl = df_entrenamiento_aux1[-trainIndex, ] # sólo las variables importantes + target

library(MLmetrics)

model_log = glm(as.factor(ATTRITION)~.,
                family=binomial(link='logit'),
                data=df_entrenamiento_rl[,-c(1,2,17,18,84,85,86)])

feature_log = names(model_log$coefficients)[-1]

pred_log = predict(model_log,
                   newdata=df_validación_rl[,-c(1,2,17,18,19,84,85,86)],
                   type='response')

LogLoss(pred_log,df_validación_rl$ATTRITION)
0.364259

# Random Forest
model_rf = randomForest(as.factor(ATTRITION)~., data = df_entrenamiento_rl[,-c(1,2,17,18,84,85,86)], ntree=100, do.trace=T, importance=T )
pred_rf = predict(model_rf,
                   newdata=df_validación_rl[,-c(1,2,17,18,19,84,85,86)],
                   type='prob')[,2]

LogLoss(pred_rf[,2],df_validación_rl$ATTRITION)
0.4189347

df_stacking = data.frame(pred_rf = pred_rf, pred_log, pred_bst)

LogLoss(apply(df_stacking, 1, mean),df_validación_rl$ATTRITION) #0.309498
LogLoss(apply(df_stacking, 1, min),df_validación_rl$ATTRITION) #0.4317772
LogLoss(apply(df_stacking, 1, max),df_validación_rl$ATTRITION) #0.3424789
options(scipen=999)
df_resul_f = data.frame(ID_CORRELATIVO = df_validación_rl$ID_CORRELATIVO,
                        pred_rf,
                        pred_log,
                        pred_bst,
                        ATTRITION_PROB = apply(df_stacking, 1, mean),
                        ATTRITION_REAL = df_validación_rl$ATTRITION
                        )
