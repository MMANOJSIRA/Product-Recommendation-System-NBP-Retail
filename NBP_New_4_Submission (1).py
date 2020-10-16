# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 19:33:29 2017

@author: Anu.Thomas
"""


from __future__ import division
import os
import time
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.preprocessing import LabelEncoder
from collections import Counter

import random
random.seed(123)

start_time = time.time()
print("\nStart time: "+ time.strftime('%Y-%m-%d %H:%M:%S') +"\n")

path =r'C:/WORK/CIP+POC+Innovation/NBP/Kaggle/' # use your path

trainDataAll = pd.read_csv(path+'train_ver2.csv')
testData = pd.read_csv(path+'test_ver2.csv')
trainDataAll = trainDataAll.append(testData, ignore_index=True)

print('File read time '+str(round((time.time() - start_time)/60,4))+" mins")

products = [ 'ind_ahor_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',  'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']    

#products = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']    

#months = np.unique(trainDataAll.fecha_dato)
#for p in products:    
#    prevCust  = np.unique(trainDataAll.loc[ np.where( (trainDataAll.fecha_dato==months[13]) & (trainDataAll[p]==1))[0], 'ncodpers'])
#    excl =  True
#  #  print p
#    for m in months[14:-1]:
#        thisCust =   np.unique(trainDataAll.loc[ np.where( (trainDataAll.fecha_dato==m) & (trainDataAll[p]==1))[0], 'ncodpers'])
#      #  print set(thisCust).difference(prevCust)
#        if len(set(thisCust).difference(prevCust)) > 0:
#            print "New customer for " + str(p) + " in " +  m
#            excl = False
#            break
#        else:
#           prevCust = thisCust
#    if excl:
#        print "Exclude " + p

#SumP = trainDataAll[products].sum(axis=1)
#Consider records with atleat one product is sold
#trainDataAll = trainDataAll.loc[list(np.where(SumP>0)[0])]
#trainDataAll = trainDataAll.query('antiguedad >0 ') # there are few records with -999999

trainDataAll.nomprov =  trainDataAll.nomprov.fillna("BLANK")
trainDataAll.segmento = trainDataAll.segmento.fillna("BLANK")

clust_columns =  ['ncodpers', 'ind_empleado','sexo','age','ind_nuevo','antiguedad', \
'indrel', 'indrel_1mes','tiprel_1mes', 'conyuemp', 'ind_actividad_cliente','renta' ,'segmento', 'indfall']

ag1 ={}
for p in products:
    ag1[p] = 'sum'

ap1 = trainDataAll.groupby(['fecha_dato']).agg(ag1)

submission =[]
trainDataAll["loopvar"] = trainDataAll.nomprov #.map(str) +" " +trainDataAll.segmento.map(str)

for prv in np.unique(trainDataAll.loopvar):
    print "running for " +  str (prv)     
    trainData =  trainDataAll[trainDataAll.loopvar==prv]
    trainData.reset_index(inplace=True)

    #prepare unique record for each customer
    custData = trainData[clust_columns].groupby('ncodpers').last()
    custData.reset_index(inplace=True)

    ###########Clearing wrong data in spouse index N=0 #############
    custData.conyuemp = [ 0 if pd.isnull(x)  else 1 for x in custData.conyuemp]
    
    custData.antiguedad = [  ( max(int(x),0)  if str(x).strip() != "NA" else 0 ) for x in custData.antiguedad]
    avgRent= np.mean([float(x) for x in custData.renta  if str(x).strip()!='NA'])
    avgAge = np.mean([int(x) for x in custData.age  if str(x).strip()!='NA'])    
    custData.age = [ int(x) if str(x).strip() != "NA"  else avgAge for x in custData.age]
    custData.renta =  [ ( avgRent if str(x).strip()=='NA' else float(x)) for x in custData.renta]
    
    custData.tiprel_1mes = [ 'M' if pd.isnull  else str(x) for x in custData.tiprel_1mes]
    custData.indrel_1mes = [ 0 if x == "P"  else (5 if pd.isnull(x)  else int(str(x).replace('.0',""))) for x in custData.indrel_1mes]
    
    
    ##########Changing Sex into numrical #########
    custData['sexo'] = custData['sexo'] .replace(['H'],1)
    custData['sexo'] = custData['sexo'] .replace(['V'],0)
    
    
    ############## Changing Tiprel_1mes ######################
    replacements = {
       'tiprel_1mes': {
          r'(A)': 1, #########active customer
          r'(I)': 0, ############# Inactive
          r'(P)': -1, ############# Former
          r'(R)': 2, ############## Potential
          r'(M)': 2 ############## Missing
       }
    }
    custData.replace(replacements, regex=True, inplace=True)
    
    ############## Changing the Employment Index of customer ind_empleado#################
    replacements = {
       'ind_empleado': {
          r'(A)': 1, #########active employee
          r'(B)': 0, ############# Ex-Employee
          r'(F)': 2, ############# Filial
          r'(N)': 3, ############## Not Employed
          r'(P)': 4,  ############## passive Employed
          r'(S)': 1 ############Given ambiguous in data. No Documentation left
               }
    }
    custData.replace(replacements, regex=True, inplace=True)
    
    # ############## Changing the indfall #################
    
    replacements = {
       'indfall': {
          r'(N)': 1, #########active employee
          r'(S)': 0 ############# Ex-Employee
       }
    }
    custData.replace(replacements, regex=True, inplace=True)
    
    
        # ############## Changing the segmento #################
    
    replacements = {
       'segmento': {
          r'(BLANK)': 0,
          r'(01 - TOP)':3,
          r'(02 - PARTICULARES)':2,
          r'(03 - UNIVERSITARIO)':1          
       }
    }
    custData.replace(replacements, regex=True, inplace=True)
    
    
    custData['indrel'] = custData['indrel'] .replace(99,0)
    
    ## Normalisation
    mina = float(min(custData.age)); maxa = float(max(custData.age))
    custData['age'] = [ (x-mina)/(maxa-mina)  for x in custData.age]
    
    mina = float(min(custData.renta)); maxa = float(max(custData.renta))
    custData['renta'] = [ 5.0*(x-mina)/(maxa-mina)  for x in custData.renta]
    
    mina = float(min(custData.antiguedad)); maxa = float(max(custData.antiguedad))
    custData['antiguedad'] = [ (x-mina)/(maxa-mina)  for x in custData.antiguedad]
    
    custData = custData.fillna(0)
  #  custData = custData[0:100]
    print('Data prep is done '+str(round((time.time() - start_time)/60,4))+" mins")
    
    #############################################################################################################
    No_of_clusters = 5;
    ##############################################################################################################
    km = KMeans(n_clusters = No_of_clusters)
    custData['cluster'] = km.fit_predict(custData[clust_columns[1:]].values)
    
    smallClust =  list(np.where(custData.cluster.value_counts() < 50)[0])
    if len(smallClust) > 1:
        custData.cluster[custData.cluster.isin(smallClust)] =  min(smallClust)
   
  #  custData.to_csv(path+'customer_cluster.csv')
    print "k-means complete"
    print custData.cluster.value_counts()
    
    
    elapsed_time = time.time() - start_time
    print('Elapsed time '+str(round(elapsed_time/60,4))+" mins")    
    
    
    for c in np.unique(custData.cluster):
        print "Cluster=",c
        custSample = custData.loc[custData.cluster ==c, 'ncodpers']
        
        trSample0 = trainData.loc[trainData['ncodpers'].isin(custSample)]
        trSample0.reset_index(inplace=True)        
        trSample0['State'] = [ "-".join([str(x)  for x in np.where(trSample0.iloc[i][products]==1)[0]]) for i in range(len(trSample0)) ]        
        rawSum = trSample0[products].sum(axis=1, keepdims=True)
        trSample =  trSample0.loc[np.where(rawSum >0)[0]] # ignore useless columns
        trSample = trSample.fillna(0)        
        
    #    trSample = trSample.query('fecha_dato!="2016-05-28"')    
        
      #  trSample = trSample.query('State!=""')
        
        AllStates =  np.unique(trSample.State)
        
        print "No of customers" ,  len(custSample)
        print "No of states" ,  len(AllStates)
        
        transMat = np.identity(len(AllStates))
        for cust in np.unique(trSample.ncodpers):            
            idx = list(trSample.loc[trSample.ncodpers==cust,'State'])    
            for ip in range(len(idx)-1):
                srcState =  list(np.where(AllStates==idx[ip])[0])[0]   
                dstState =  list(np.where(AllStates==idx[ip+1])[0])[0]   
                transMat[srcState,dstState] = transMat[srcState,dstState]+1
        
        transMat = transMat/transMat.sum(axis=1, keepdims=True)
        transMat=pd.DataFrame(transMat)
        AllStates = ["S-"+str(x) for x in AllStates]
        transMat.columns =  AllStates
        transMat.index =  AllStates
        for i in range(len(transMat)):
            transMat.iloc[i,i] = -1
        
        temp = [ list(np.argsort(transMat.iloc[i,]))[::-1] for i in range(len(transMat))]
        temp =  [ x[0:min(len(x),3)]  for x in temp]
        temp = [ [x1 for x1 in temp[i] if transMat.iloc[i,x1]>0] for i in range(len(temp))]
        transMat['NextState'] = [[AllStates[u]  for u in x]  for x in temp]
  
     # transMat.to_csv(path+'TransMatrix_'+str(c)+'.csv')
        transMat['CurrState'] = AllStates
        clsRecommend = transMat[['CurrState','NextState']]
        
        print('Elapsed time '+str(round((time.time() - start_time)/60,4))+" mins")
    
        valData = trSample0.query('fecha_dato=="2016-06-28"')[['ncodpers']]
        valDataMay = trSample0.query('fecha_dato=="2016-05-28"')[['ncodpers','State']]
        valData.reset_index(inplace=True);valDataMay.reset_index(inplace=True);
        valData =  pd.merge(valData,valDataMay, on='ncodpers', how='left' )
        valData['CurrState'] = ["S-"+("" if pd.isnull(x) else x) for x in valData.State]
        NewData1 = pd.merge(valData[['ncodpers','CurrState']], clsRecommend, on=['CurrState'],how='left')
        
        NewData1['added_products'] = [""  for i in range(len(NewData1)) ]
        NewData1['AssignFlag'] = ["Default"  for i in range(len(NewData1)) ]
        pop = trSample[products].sum(axis=0)
        pop.index =  [str(x) for x in range(len(pop))]
        pop.sort_values(ascending=False, inplace=True)
        for i in range(len(NewData1)):
            a=  set(str(NewData1.CurrState[i]).replace("S-","").split("-"))
            prod_list = []
            if str(NewData1.NextState[i])!='nan':
                b0 = NewData1.NextState[i]
                for xb in b0:
                    b = set(str(xb).replace("S-","").split("-"))
                    prd1  = b.difference(a).difference(['']) 
                    for pdt in prd1:
                        if len(prod_list) > 6:
                            break
                        if products[int(pdt)] not in prod_list:
                            prod_list.append(products[int(pdt)])
            aFlag = "Trans"  if len(prod_list) >0 else "Popular"            
            for x in pop.index:
                if x not in a:
                    if len(prod_list) > 6:
                            break
                    if products[int(x)] not in prod_list:
                        prod_list.append(products[int(x)])

            NewData1.set_value(i, 'AssignFlag', aFlag)
            NewData1.set_value(i, 'added_products', str(" ".join(prod_list) )   )
         #   print c, i, NewData1.added_products[i]
    
        NewData1['cluster'] = c
        NewData1['province'] =  str (prv) 
        if len(submission)==0:
            submission =  NewData1
        else:
            submission = submission.append(NewData1)
            


submission[['ncodpers', 'added_products']].to_csv(path+'Submission.csv', index=False)
submission.to_csv(path+'Submission_full_new.csv', index=False)

elapsed_time = time.time() - start_time
print("\nEnd time: "+ time.strftime('%Y-%m-%d %H:%M:%S'))
print('Elapsed time '+str(round(elapsed_time/60,4))+" mins")



