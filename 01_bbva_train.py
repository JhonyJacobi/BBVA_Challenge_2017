# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:05:31 2017

@author: jjacobir
"""
import gc
import os
import pandas as pd
#from dplython import (DplyFrame, X, diamonds, select, sift, sample_n,
#    sample_frac, head, arrange, mutate, group_by, summarize, DelayFunction)
from pandas_ply import install_ply, X, sym_call
install_ply(pd)
#from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
#import numpy as np
import random
import xgboost as xgb 
#from matplotlib import pyplot

gc.enable()

os.getcwd()
os.chdir("D:/Competencia/BBVA")

# Cargar Data
data_train_aux = pd.read_excel("01_Data/train_clientes.xlsx")
data_train_req_aux = pd.read_excel("01_Data/train_requerimientos.xlsx")

data_train_req_aux.head()
type(data_train_aux)

data_train_aux.groupby("ATTRITION")["ID_CORRELATIVO"].count()

# Vista de la data
data_train_aux.describe().transpose()
data_train_aux.info()
data_train_aux.dtypes

# % de missing 
data_train_aux.isnull().sum()/len(data_train_aux)
data_train_req_aux.isnull().sum()/len(data_train_req_aux)

# Crear variaable número de na's
data_train_aux["NRO_NA"] = data_train_aux.isnull().sum(axis=1)

# Crear dummies
data_train_dumm = pd.get_dummies(data_train_aux,
                         columns = ['RANG_INGRESO','FLAG_LIMA_PROVINCIA','RANG_SDO_PASIVO_MENOS0','RANG_NRO_PRODUCTOS_MENOS0'], 
                         prefix = ['DUMM_RANG_INGRESO','DUMM_FLAG_LIMA_PROVINCIA','DUMM_RANG_SDO_PASIVO_MENOS0','DUMM_RANG_NRO_PRODUCTOS_MENOS0']
                         )

data_train_dumm = data_train_dumm.set_index('ID_CORRELATIVO')

# Crear variables transaccionales
"""
data_train_req_aux = DplyFrame(data_train_req_aux)

data_train_req_aux_ag = (data_train_req_aux >> 
  group_by(X.ID_CORRELATIVO) >> 
  summarize(REQ_INI_MES=X.CODMES.min(),
            REQ_ULT_MES=X.CODMES.max(),
            REQ_NUM_PROD = X.PRODUCTO_SERVICIO_2.nunique(),
            REQ_NUM_SUBMOT = X.SUBMOTIVO_2.nunique(),
            #REQ_NUM_REC = sum(X.TIPO_REQUERIMIENTO2 == 'Reclamo')#,
                   #REQ_NUM_SOL = sum(TIPO_REQUERIMIENTO2 == "Solicitud"),
                   #REQ_NUM_PROC_TOT = sum(DICTAMEN == "PROCEDE TOTAL"),
                   #REQ_NUM_PROC_PAR = sum(DICTAMEN == "PROCEDE PARCIAL"),
                   #REQ_NUM_PROC_NOP = sum(DICTAMEN == "NO PROCEDE"),
            REQ_NUM_TOT = X.CODMES.count()
            ) #>> head(10)
)
"""

data_train_req_aux_ag = (data_train_req_aux
                         .groupby(['ID_CORRELATIVO'])
                         .ply_select(REQ_INI_MES = X.CODMES.min(),
                                     REQ_ULT_MES = X.CODMES.max(),
                                     REQ_NUM_PROD = X.PRODUCTO_SERVICIO_2.nunique(),
                                     REQ_NUM_SUBMOT = X.SUBMOTIVO_2.nunique(),
                                     REQ_NUM_REC = bool([X.TIPO_REQUERIMIENTO2 == 'Solicitud']),
                                     REQ_NUM_TOT = X.CODMES.count())
                         )#.head(20)


data_train_req_aux_ag = pd.DataFrame(data_train_req_aux_ag, 
                                     index = data_train_req_aux_ag['ID_CORRELATIVO'],
                                     dtype = {'ID_CORRELATIVO':'int',
                                              'REQ_INI_MES':'object',
                                              'REQ_NUM_PROD':'int',
                                              'REQ_NUM_SUBMOT':'int',
                                              'REQ_NUM_TOT':'int',
                                              'REQ_ULT_MES':'object'}
                                     )

# Agregar variables de requerimientos

data_train_dumm .shape
data_train_req_aux_ag.shape

data_train_tot = pd.concat([data_train_dumm,data_train_req_aux_ag], axis=1, join='outer')

data_train_tot.isnull().sum()

# filter
(data_train_tot
  .ply_where(X.ID_CORRELATIVO == 3))

# Crear data de entrenamiento y validación
random.seed(123)
train, val = train_test_split(data_train_tot, test_size=0.2)

# Modelo Xgboost
#y_train = train.pop('ATTRITION')
#y_val = val.pop('ATTRITION')

dtrain = xgb.DMatrix(data = train.drop(['CODMES','ATTRITION','REQ_ULT_MES','REQ_INI_MES'], axis = 1), label=train['ATTRITION'] )
dval = xgb.DMatrix(data = val.drop(['CODMES','ATTRITION','REQ_ULT_MES','REQ_INI_MES'], axis = 1), label=val['ATTRITION'] )

param = {'max_depth':9, 
         'eta':0.06, 
         'silent':1, 
         'objective':'binary:logistic',
         'eval_metric':'logloss',
         'min_child_weight':10,
         'booster': 'gbtree'
         }
num_round = 200

watchlist = [(dval, 'eval'), (dtrain, 'train')]

random.seed(123)
bst = xgb.train(param, 
                dtrain, 
                num_round,
                watchlist
                )

pd.DataFrame(bst.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)


xgb.plot_importance(bst, max_num_features=30)
show()


