#RS code implements random forest algorithm on given datasets. Test model under executables.  
import sklearn
from sklearn import metrics
from numpy import mean,std,cov
import pandas as pd
from sklearn.linear_model import ElasticNet,Lasso, Ridge,LinearRegression
from sklearn.model_selection import cross_val_score,RepeatedKFold,train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from astropy.table import Table,Column
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib import cm 
import deepdish as dd
import hdf_reader as hdf

dir_TNG='/x/Physics/AstroPhysics/Shared-New/DATA/IllustrisTNG/'
my_dir='/home/AstroPhysics-Shared/Sharma/Illustris/data_output/'
plot_dir='/home/AstroPhysics-Shared/Sharma/Illustris/plots/'
my_imgs='/home/AstroPhysics-Shared/Sharma/Illustris/images/'
output_dir=dir_TNG+'/bulgegrowth/output/'

model_list={'Random_Forest':RandomForestRegressor(),'ElasticNet':ElasticNet(),'Lasso':Lasso(),'Ridge':Ridge(),'LinearRegression':LinearRegression()}
hyperparam_dis={'n_estimators':range(10,100),'max_depth':range(3,10)}

def evaluate_model(X,y,model_name,splits,repeats):
    #use regressor model to generate new model from data X and outcome y. search cross validator to get best hyperparams. Train and test sample
    model=RandomForestRegressor()
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=11)
    print('got training and test sets')
    #print(X_train,y_train)
    all_datasets=[X_train,X_test]
    for data in all_datasets:
        for ncol in data.columns:
            #print(ncol,data[ncol])
            median=np.nanmedian(data[ncol])
            #print(ncol,median)
            data[ncol].fillna(median,inplace=True)
    search=RandomizedSearchCV(estimator=model,n_iter=5,param_distributions=hyperparam_dis,random_state=11)
    search.fit(X=X_train,y=y_train)
    params=search.best_params_
    MAE_score=search.score(X_test,y_test)
    cv=RepeatedKFold(n_splits=splits,n_repeats=repeats,random_state=11)
    #print(cv)
    model=RandomForestRegressor(n_estimators=params['n_estimators'],max_depth=params['max_depth'])
    fitted_model=model.fit(X=X_train,y=y_train)
    n_scores=cross_val_score(fitted_model,X=X_test,y=y_test,cv=cv,n_jobs=-1,error_score='raise',scoring='r2')
    R2_mean=n_scores.mean()
    R2_std=n_scores.std()
    #print(n_scores)
    #print('MAE: %.3f(%.3f)'%(mean(n_scores),std(n_scores)))
    return(R2_mean,R2_std,fitted_model,MAE_score,X_test,y_test,y_train,X_train)

def predict_data(X_test,model):
    yhat=model.predict(X_test)
    #print(yhat)
    return(yhat)

                                                                                                      
####-------Execute Code----------------------######################################
#test different estimators
#column hdrs are 'ID','log_stellar_mass','BT_ratio','half_stellar_radius','log_SFR','log_sSFR','CAS_clump_idx','CAS_asym_idx','CAS_conc_idx','Blob_idx','Leaf_idx'
#read the csv into datframes. y is outcome, X is array of variables
#SFGz2=pd.read_csv(my_dir+'SFGs_10to115_at_z2_data',header=0)
#SFGz1=pd.read_csv(my_dir+'SFGs_10to115_at_z1_data',header=0)
#descendantsz1=pd.read_csv(my_dir+'descendants_at_z1_data',header=0)
'''
combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
del combined_data['subfindID']
del combined_data['ID']
del combined_data['BT_ratio']
del combined_data['Unnamed: 0']
for i in range(67):
    del combined_data['all_BT'+str(i)]
    del combined_data['lmbulge'+str(i)]
combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']>0.1].index)
del combined_data['summed_merger_ratio']

y=combined_data['d_BT_2.0']
del combined_data['d_BT_0.5']
del combined_data['d_BT_1.0']
del combined_data['d_BT_2.0']
del combined_data['d_lmbulge_0.5']
del combined_data['d_lmbulge_1.0']
del combined_data['d_lmbulge_2.0']
del combined_data['rel_BT_del_0.2']
del combined_data['rel_BT_del_0.1']
del combined_data['abs_BT_del_0.3']
del combined_data['abs_BT_del_0.2']
all_col=combined_data.columns
X=combined_data
i=0

model=model_list['Random_Forest']
result=evaluate_model(X=X,y=y,model_name=model,splits=2,repeats=10)
fitted_model=result[2]
y_predict=predict_data(X_test=result[4],model=fitted_model)
y_actual=result[5]
X_test=result[4]
y_train=result[6]
score=result[0]
score_sd=result[1]
print(f'score is {score} with sd {score_sd} and MAE {result[3]}')
y_train_predict=predict_data(X_test=result[7],model=fitted_model)
fig,ax=plt.subplots(figsize=(8,4))
ax.scatter(y_actual,y_predict,c='black',s=4,label='Test values')
ax.scatter(y_train,y_train_predict,c='darkgray',s=4,label='Training values')
x=np.linspace(np.min(y_actual),np.max(y_actual),len(y_actual))
y=x
ax.set_xlim((np.min(y_actual),np.max(y_actual)))
ax.set_ylim((np.min(y_predict),np.max(y_predict)))
ax.plot(x,y,color='k',lw=0.5,ls='-')
ax.set_xlabel('Real BT growth over 2 Gyr')
ax.set_ylabel('Predicted BT growth over 2 Gyr')
plt.legend()
#plt.show()
fig.savefig(plot_dir+'combined_data_model_2')
print(f'R_2 is {result[0]}'+f' and MAE is {result[3]}')  
r=permutation_importance(fitted_model,X_test,y_actual,n_repeats=30,random_state=11)
#spits out mean importance as first array and std importance as second. 
idx=np.argsort(r.importances_mean)
print(len(idx))
fig,ax=plt.subplots()
all_imp_vals=[]
all_cols=[]
sorted_col=all_col[idx]
for i in idx:
    #print(f'i is {i}')
    col_name=all_col[i]
    all_cols.append(col_name)
    #print(col_name)
    imp_values=r.importances[i,:]
    #print(imp_values)
    all_imp_vals.append(imp_values)
print(all_imp_vals)    
ax.boxplot(all_imp_vals,labels=all_cols,vert=0,showfliers=False)
ax.set_xlabel('Permutation based importance')
plt.show()
fig.savefig(plot_dir+'permutation_importance_plot_2gyr',dpi=600)
sorted_col=all_col[idx]
sorted_mean_imp=r.importances_mean[idx]
sorted_sd_imp=r.importances_std[idx]
#print(r.importances.shape)
        
#print(r)
#for  i in r.importances+mean.argsort()[::-1]:
    
#importance=fitted_model.feature_importances_
#print(importance)
#idx=np.argsort(importance)
#sorted_imp=importance[idx]
#sorted_col=all_col[idx]
#print(sorted_imp,sorted_col)
#fig,ax=plt.subplots()
#ax.barh(sorted_col,sorted_imp)
#ax.set_xlabel('Impurity-based Importance') 
#plt.show()
#fig.savefig(plot_dir+'combined_importance_plot_1gyr',dpi=600)
'''
'''
#get all merger poor permutations importance and predictive models onto one plot for poster comment out bits you dont need
outcome_list=['d_BT_0.5','d_BT_2.0']
label_list=['0.5 Gyr','2 Gyr']
nrows=2
ncols=2
fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(12,12))
'''
'''
for i,outcome in enumerate(outcome_list):
    combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
    del combined_data['subfindID']
    del combined_data['ID']
    del combined_data['BT_ratio']
    del combined_data['Unnamed: 0']
    for j in range(67):
        del combined_data['all_BT'+str(j)]
        del combined_data['lmbulge'+str(j)]
    combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']>0.01].index)
    del combined_data['summed_merger_ratio']
    y=combined_data[outcome]
    #removing future data and anything that is colinear prior to building teh model and extracting importance. Decided to go with v squared fractions and mass fractions and take vphi and vr seperately. 
    del combined_data['d_BT_0.5']
    del combined_data['d_BT_1.0']
    del combined_data['d_BT_2.0']
    del combined_data['d_lmbulge_0.5']
    del combined_data['d_lmbulge_1.0']
    del combined_data['d_lmbulge_2.0']
    del combined_data['rel_BT_del_0.2']
    del combined_data['rel_BT_del_0.1']
    del combined_data['abs_BT_del_0.3']
    del combined_data['abs_BT_del_0.2']
    del combined_data['lMgas']
    del combined_data['lMcoldgas']
    del combined_data['star_avg_vphiv']
    del combined_data['star_avg_vRv']
    del combined_data['star_avg_vzv']
    del combined_data['gas_avg_vphiv']
    del combined_data['gas_avg_vRv']
    del combined_data['gas_avg_vzv']
    del combined_data['CAS_clump_idx']
    del combined_data['Leaf_idx']
    #delete the following parapeters depending on interest in direction
    del combined_data['gas_avg_v2zv2']
    #del combined_data['gas_avg_v2Rv2']
    del combined_data['gas_avg_v2phiv2']    
    del combined_data['star_avg_v2zv2']
    #del combined_data['star_avg_v2Rv2']
    del combined_data['star_avg_v2phiv2']
    all_col=combined_data.columns
    X=combined_data
    model=model_list['Random_Forest']
    result=evaluate_model(X=X,y=y,model_name=model,splits=2,repeats=10)
    fitted_model=result[2]
    y_predict=predict_data(X_test=result[4],model=fitted_model)
    y_actual=result[5]
    X_test=result[4]
    y_train=result[6]
    score=result[0]
    score_sd=result[1]
    print(f'score is {score} with sd {score_sd} and MAE {result[3]}')
    y_train_predict=predict_data(X_test=result[7],model=fitted_model)
    correlation_test=(pearsonr(y_predict,y_actual)[0])**2
    correlation_train=(pearsonr(y_train_predict,y_train)[0])**2
    correlation_test=round(correlation_test,3)
    correlation_train=round(correlation_train,3)
    ax[0,i].scatter(y_actual,y_predict,c='black',s=4,label='Test values')
    ax[0,i].scatter(y_train,y_train_predict,c='darkgray',s=4,label='Training values')
    ax[0,i].legend(fontsize=10,labelspacing=0.2,columnspacing=1,loc='upper left')
    x=np.linspace(np.min(y_actual),np.max(y_actual),len(y_actual))
    y=x
    ax[0,i].set_xlim((np.min(y_actual),np.max(y_actual)))
    ax[0,i].set_ylim((np.min(y_predict),np.max(y_predict)))
    ax[0,i].plot(x,y,color='k',lw=0.5,ls='-')
    ax[0,i].text(np.min(y_actual),np.max(y_predict),r'Test data $R^2=$'+str(correlation_test),fontsize=14)
    ax[0,i].text(np.max(y_actual),np.max(y_predict),r'Training data $R^2=$'+str(correlation_train),fontsize=14,ha='right')
    ax[0,i].set_xlabel('Real BT growth over '+label_list[i],fontsize=14)
    ax[0,i].set_ylabel('Predicted BT growth over ' +label_list[i],fontsize=14)
    #ax[0,i].set_title('Quiet Merger Galaxies',fontsize=10)
    #print(f'R_2 is {result[0]}'+f' and MAE is {result[3]}')  
    r=permutation_importance(fitted_model,X_test,y_actual,n_repeats=30,random_state=11)
#spits out mean importance as first array and std importance as second. 
    idx=np.argsort(r.importances_mean)
    print(len(idx))
    all_imp_vals=[]
    all_cols=[]
    sorted_col=all_col[idx]
    for k in idx:
        #print(f'i is {i}')
        col_name=all_col[k]
        all_cols.append(col_name)
        #print(col_name)
        imp_values=r.importances[k,:]
        #print(imp_values)
        all_imp_vals.append(imp_values)
    print(all_imp_vals)    
    ax[1,i].boxplot(all_imp_vals,vert=0,showfliers=False)
    ax[1,i].set_yticklabels(all_cols,fontsize=14)
    ax[1,i].set_xlabel('Permutation based importance for growth over '+label_list[i],fontsize=14)
    ax[1,i].set_title(r'Only $v^2R/v^2$ considered',fontsize=14) 
#now do vphi
'''
'''
for i,outcome in enumerate(outcome_list):
    combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
    del combined_data['subfindID']
    del combined_data['ID']
    del combined_data['BT_ratio']
    del combined_data['Unnamed: 0']
    for j in range(67):
        del combined_data['all_BT'+str(j)]
        del combined_data['lmbulge'+str(j)]
    combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']>0.01].index)
    del combined_data['summed_merger_ratio']
    y=combined_data[outcome]
    #removing future data and anything that is colinear prior to building teh model and extracting importance. Decided to go with v squared fractions and mass fractions and take vphi and vr seperately. 
    del combined_data['d_BT_0.5']
    del combined_data['d_BT_1.0']
    del combined_data['d_BT_2.0']
    del combined_data['d_lmbulge_0.5']
    del combined_data['d_lmbulge_1.0']
    del combined_data['d_lmbulge_2.0']
    del combined_data['rel_BT_del_0.2']
    del combined_data['rel_BT_del_0.1']
    del combined_data['abs_BT_del_0.3']
    del combined_data['abs_BT_del_0.2']
    del combined_data['lMgas']
    del combined_data['lMcoldgas']
    del combined_data['star_avg_vphiv']
    del combined_data['star_avg_vRv']
    del combined_data['star_avg_vzv']
    del combined_data['gas_avg_vphiv']
    del combined_data['gas_avg_vRv']
    del combined_data['gas_avg_vzv']
    del combined_data['CAS_clump_idx']
    del combined_data['Leaf_idx']
    #delete the following parapeters depending on interest in direction
    del combined_data['gas_avg_v2zv2']
    del combined_data['gas_avg_v2Rv2']
    #del combined_data['gas_avg_v2phiv2']    
    del combined_data['star_avg_v2zv2']
    del combined_data['star_avg_v2Rv2']
    #del combined_data['star_avg_v2phiv2']
    all_col=combined_data.columns
    X=combined_data
    model=model_list['Random_Forest']
    result=evaluate_model(X=X,y=y,model_name=model,splits=2,repeats=10)
    fitted_model=result[2]
    y_predict=predict_data(X_test=result[4],model=fitted_model)
    y_actual=result[5]
    X_test=result[4]
    y_train=result[6]
    score=result[0]
    score_sd=result[1]
    print(f'score is {score} with sd {score_sd} and MAE {result[3]}')
    y_train_predict=predict_data(X_test=result[7],model=fitted_model)
    correlation_test=(pearsonr(y_predict,y_actual)[0])**2
    correlation_train=(pearsonr(y_train_predict,y_train)[0])**2
    correlation_test=round(correlation_test,3)
    correlation_train=round(correlation_train,3)
    ax[0,i].scatter(y_actual,y_predict,c='black',s=4,label='Test values')
    ax[0,i].scatter(y_train,y_train_predict,c='darkgray',s=4,label='Training values')
    ax[0,i].legend(fontsize=10,labelspacing=0.2,columnspacing=1,loc='upper left')
    x=np.linspace(np.min(y_actual),np.max(y_actual),len(y_actual))
    y=x
    ax[0,i].set_xlim((np.min(y_actual),np.max(y_actual)))
    ax[0,i].set_ylim((np.min(y_predict),np.max(y_predict)))
    ax[0,i].plot(x,y,color='k',lw=0.5,ls='-')
    ax[0,i].text(np.min(y_actual),np.max(y_predict),r'Test data $R^2=$'+str(correlation_test),fontsize=14)
    ax[0,i].text(np.max(y_actual),np.max(y_predict),r'Training data $R^2=$'+str(correlation_train),fontsize=14,ha='right')
    ax[0,i].set_xlabel('Real BT growth over '+label_list[i],fontsize=14)
    ax[0,i].set_ylabel('Predicted BT growth over ' +label_list[i],fontsize=14)
    #ax[0,i+2].set_title('Quiet Merger Galaxies',fontsize=10)
    #print(f'R_2 is {result[0]}'+f' and MAE is {result[3]}')  
    r=permutation_importance(fitted_model,X_test,y_actual,n_repeats=30,random_state=11)
#spits out mean importance as first array and std importance as second. 
    idx=np.argsort(r.importances_mean)
    print(len(idx))
    all_imp_vals=[]
    all_cols=[]
    sorted_col=all_col[idx]
    for k in idx:
        #print(f'i is {i}')
        col_name=all_col[k]
        all_cols.append(col_name)
        #print(col_name)
        imp_values=r.importances[k,:]
        #print(imp_values)
        all_imp_vals.append(imp_values)
    print(all_imp_vals)    
    ax[1,i].boxplot(all_imp_vals,vert=0,showfliers=False)
    ax[1,i].set_yticklabels(all_cols,fontsize=14)
    ax[1,i].set_xlabel('Permutation based importance for growth over '+label_list[i],fontsize=14)
    ax[1,i].set_title(r'Only $v^2phi/v^2$ considered',fontsize=14)

'''
'''
#now do same for high merger galaxies    
for i,outcome in enumerate(outcome_list):
    combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
    del combined_data['subfindID']
    del combined_data['ID']
    del combined_data['BT_ratio']
    del combined_data['Unnamed: 0']
    for j in range(67):
        del combined_data['all_BT'+str(j)]
        del combined_data['lmbulge'+str(j)]
    combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']<0.01].index)
    del combined_data['summed_merger_ratio']
    y=combined_data[outcome]
    #removing future data and anything that is colinear prior to building teh model and extracting importance. Decided to go with v squared fractions and mass fractions and take vphi and vr seperately. 
    del combined_data['d_BT_0.5']
    del combined_data['d_BT_1.0']
    del combined_data['d_BT_2.0']
    del combined_data['d_lmbulge_0.5']
    del combined_data['d_lmbulge_1.0']
    del combined_data['d_lmbulge_2.0']
    del combined_data['rel_BT_del_0.2']
    del combined_data['rel_BT_del_0.1']
    del combined_data['abs_BT_del_0.3']
    del combined_data['abs_BT_del_0.2']
    del combined_data['lMgas']
    del combined_data['lMcoldgas']
    del combined_data['star_avg_vphiv']
    del combined_data['star_avg_vRv']
    del combined_data['star_avg_vzv']
    del combined_data['gas_avg_vphiv']
    del combined_data['gas_avg_vRv']
    del combined_data['gas_avg_vzv']
    del combined_data['CAS_clump_idx']
    del combined_data['Leaf_idx']
    #delete the following parapeters depending on interest in direction
    del combined_data['gas_avg_v2zv2']
    #del combined_data['gas_avg_v2Rv2']
    del combined_data['gas_avg_v2phiv2']    
    del combined_data['star_avg_v2zv2']
    #del combined_data['star_avg_v2Rv2']
    del combined_data['star_avg_v2phiv2']
    all_col=combined_data.columns
    X=combined_data
    model=model_list['Random_Forest']
    result=evaluate_model(X=X,y=y,model_name=model,splits=2,repeats=10)
    fitted_model=result[2]
    y_predict=predict_data(X_test=result[4],model=fitted_model)
    y_actual=result[5]
    X_test=result[4]
    y_train=result[6]
    score=result[0]
    score_sd=result[1]
    print(f'score is {score} with sd {score_sd} and MAE {result[3]}')
    y_train_predict=predict_data(X_test=result[7],model=fitted_model)
    correlation_test=(pearsonr(y_predict,y_actual)[0])**2
    correlation_train=(pearsonr(y_train_predict,y_train)[0])**2
    correlation_test=round(correlation_test,3)
    correlation_train=round(correlation_train,3)
    ax[0,i].scatter(y_actual,y_predict,c='black',s=4,label='Test values')
    ax[0,i].scatter(y_train,y_train_predict,c='darkgray',s=4,label='Training values')
    ax[0,i].legend(fontsize=10,labelspacing=0.2,columnspacing=1,loc='upper left')
    x=np.linspace(np.min(y_actual),np.max(y_actual),len(y_actual))
    y=x
    ax[0,i].set_xlim((np.min(y_actual),np.max(y_actual)))
    ax[0,i].set_ylim((np.min(y_predict),np.max(y_predict)))
    ax[0,i].plot(x,y,color='k',lw=0.5,ls='-')
    ax[0,i].text(np.min(y_actual),np.max(y_predict),r'Test data $R^2=$'+str(correlation_test),fontsize=10)
    ax[0,i].text(np.max(y_actual),np.max(y_predict),r'Training data $R^2=$'+str(correlation_train),fontsize=10,ha='right')
    ax[0,i].set_xlabel('Real BT growth over '+label_list[i],fontsize=10)
    ax[0,i].set_ylabel('Predicted BT growth over ' +label_list[i],fontsize=10)
    ax[0,i].set_title('High Merger Galaxies',fontsize=10)
    #print(f'R_2 is {result[0]}'+f' and MAE is {result[3]}')  
    r=permutation_importance(fitted_model,X_test,y_actual,n_repeats=30,random_state=11)
#spits out mean importance as first array and std importance as second. 
    idx=np.argsort(r.importances_mean)
    print(len(idx))
    all_imp_vals=[]
    all_cols=[]
    sorted_col=all_col[idx]
    for k in idx:
        #print(f'i is {i}')
        col_name=all_col[k]
        all_cols.append(col_name)
        #print(col_name)
        imp_values=r.importances[k,:]
        #print(imp_values)
        all_imp_vals.append(imp_values)
    print(all_imp_vals)    
    ax[1,i].boxplot(all_imp_vals,vert=0,showfliers=False)
    ax[1,i].set_yticklabels(all_cols,fontsize=10)
    ax[1,i].set_xlabel('Permutation based importance for growth over '+label_list[i],fontsize=10)
    ax[1,i].set_title(r'Only $v^2R/v^2$ considered',fontsize=10) 
'''
'''
#now do vphi
for i,outcome in enumerate(outcome_list):
    combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
    del combined_data['subfindID']
    del combined_data['ID']
    del combined_data['BT_ratio']
    del combined_data['Unnamed: 0']
    for j in range(67):
        del combined_data['all_BT'+str(j)]
        del combined_data['lmbulge'+str(j)]
    combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']>0.01].index)
    del combined_data['summed_merger_ratio']
    y=combined_data[outcome]
    #removing future data and anything that is colinear prior to building teh model and extracting importance. Decided to go with v squared fractions and mass fractions and take vphi and vr seperately. 
    del combined_data['d_BT_0.5']
    del combined_data['d_BT_1.0']
    del combined_data['d_BT_2.0']
    del combined_data['d_lmbulge_0.5']
    del combined_data['d_lmbulge_1.0']
    del combined_data['d_lmbulge_2.0']
    del combined_data['rel_BT_del_0.2']
    del combined_data['rel_BT_del_0.1']
    del combined_data['abs_BT_del_0.3']
    del combined_data['abs_BT_del_0.2']
    del combined_data['lMgas']
    del combined_data['lMcoldgas']
    del combined_data['star_avg_vphiv']
    del combined_data['star_avg_vRv']
    del combined_data['star_avg_vzv']
    del combined_data['gas_avg_vphiv']
    del combined_data['gas_avg_vRv']
    del combined_data['gas_avg_vzv']
    del combined_data['CAS_clump_idx']
    del combined_data['Leaf_idx']
    #delete the following parapeters depending on interest in direction
    del combined_data['gas_avg_v2zv2']
    del combined_data['gas_avg_v2Rv2']
    #del combined_data['gas_avg_v2phiv2']    
    del combined_data['star_avg_v2zv2']
    del combined_data['star_avg_v2Rv2']
    #del combined_data['star_avg_v2phiv2']
    all_col=combined_data.columns
    X=combined_data
    model=model_list['Random_Forest']
    result=evaluate_model(X=X,y=y,model_name=model,splits=2,repeats=10)
    fitted_model=result[2]
    y_predict=predict_data(X_test=result[4],model=fitted_model)
    y_actual=result[5]
    X_test=result[4]
    y_train=result[6]
    score=result[0]
    score_sd=result[1]
    print(f'score is {score} with sd {score_sd} and MAE {result[3]}')
    y_train_predict=predict_data(X_test=result[7],model=fitted_model)
    correlation_test=(pearsonr(y_predict,y_actual)[0])**2
    correlation_train=(pearsonr(y_train_predict,y_train)[0])**2
    correlation_test=round(correlation_test,3)
    correlation_train=round(correlation_train,3)
    ax[0,i].scatter(y_actual,y_predict,c='black',s=4,label='Test values')
    ax[0,i].scatter(y_train,y_train_predict,c='darkgray',s=4,label='Training values')
    ax[0,i].legend(fontsize=10,labelspacing=0.2,columnspacing=1,loc='upper left')
    x=np.linspace(np.min(y_actual),np.max(y_actual),len(y_actual))
    y=x
    ax[0,i].set_xlim((np.min(y_actual),np.max(y_actual)))
    ax[0,i].set_ylim((np.min(y_predict),np.max(y_predict)))
    ax[0,i].plot(x,y,color='k',lw=0.5,ls='-')
    ax[0,i].text(np.min(y_actual),np.max(y_predict),r'Test data $R^2=$'+str(correlation_test),fontsize=10)
    ax[0,i].text(np.max(y_actual),np.max(y_predict),r'Training data $R^2=$'+str(correlation_train),fontsize=10,ha='right')
    ax[0,i].set_xlabel('Real BT growth over '+label_list[i],fontsize=10)
    ax[0,i].set_ylabel('Predicted BT growth over ' +label_list[i],fontsize=10)
    #ax[0,i].set_title('High Merger Galaxies',fontsize=10)
    #print(f'R_2 is {result[0]}'+f' and MAE is {result[3]}')  
    r=permutation_importance(fitted_model,X_test,y_actual,n_repeats=30,random_state=11)
#spits out mean importance as first array and std importance as second. 
    idx=np.argsort(r.importances_mean)
    print(len(idx))
    all_imp_vals=[]
    all_cols=[]
    sorted_col=all_col[idx]
    for k in idx:
        #print(f'i is {i}')
        col_name=all_col[k]
        all_cols.append(col_name)
        #print(col_name)
        imp_values=r.importances[k,:]
        #print(imp_values)
        all_imp_vals.append(imp_values)
    print(all_imp_vals)    
    ax[1,i].boxplot(all_imp_vals,vert=0,showfliers=False)
    ax[1,i].set_yticklabels(all_cols,fontsize=10)
    ax[1,i].set_xlabel('Permutation based importance for growth over '+label_list[i],fontsize=10)
    ax[1,i].set_title(r'Only $v^2phi/v^2$ considered',fontsize=10)
'''
'''
plt.tight_layout()
plt.show()
fig.savefig(plot_dir+'all_model_importance_plots_highmergersphi.png',dpi=600,format='png')
'''

'''
#compare importance plots for low merger vs all merger at all 3 bt_growth
outcome_list=['d_BT_0.5','d_BT_1.0','d_BT_2.0']
label_list=['0.5 Gyr','1 Gyr','2 Gyr']
nrows=1
ncols=3
fig,ax=plt.subplots(nrows=nrows,ncols=ncols,figsize=(9,3))
for i,outcome in enumerate(outcome_list):
    combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
    del combined_data['subfindID']
    del combined_data['ID']
    del combined_data['BT_ratio']
    del combined_data['Unnamed: 0']
    for j in range(67):
        del combined_data['all_BT'+str(j)]
        del combined_data['lmbulge'+str(j)]
    combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']>0.1].index)
    del combined_data['summed_merger_ratio']
    y=combined_data[outcome]
    del combined_data['d_BT_0.5']
    del combined_data['d_BT_1.0']
    del combined_data['d_BT_2.0']
    del combined_data['d_lmbulge_0.5']
    del combined_data['d_lmbulge_1.0']
    del combined_data['d_lmbulge_2.0']
    del combined_data['rel_BT_del_0.2']
    del combined_data['rel_BT_del_0.1']
    del combined_data['abs_BT_del_0.3']
    del combined_data['abs_BT_del_0.2']
    all_col=combined_data.columns
    X=combined_data
    model=model_list['Random_Forest']
    result=evaluate_model(X=X,y=y,model_name=model,splits=2,repeats=10)
    fitted_model=result[2]
    y_predict=predict_data(X_test=result[4],model=fitted_model)
    y_actual=result[5]
    correlation=(pearsonr(y_predict,y_actual)[0])**2
    X_test=result[4]
    y_train=result[6]
    score=result[0]
    score_sd=result[1]
    print(f'score is {score} with sd {score_sd} and MAE {result[3]}')
    y_train_predict=predict_data(X_test=result[7],model=fitted_model)
    model=model_list['Random_Forest']
    result=evaluate_model(X=X,y=y,model_name=model,splits=2,repeats=10)
    fitted_model=result[2]
    #print(f'R_2 is {result[0]}'+f' and MAE is {result[3]}')  
    #r=permutation_importance(fitted_model,X_test,y_actual,n_repeats=30,random_state=11)
#spits out mean importance as first array and std importance as second. 
    idx=np.argsort(r.importances_mean)
    print(len(idx))
    all_imp_vals=[]
    all_cols=[]
    sorted_col=all_col[idx]
    for k in idx:
        #print(f'i is {i}')
        col_name=all_col[k]
        all_cols.append(col_name)
        #print(col_name)
        imp_values=r.importances[k,:]
        #print(imp_values)
        all_imp_vals.append(imp_values)
    print(all_imp_vals)    
    ax[i].boxplot(all_imp_vals,labels=all_cols,vert=0,showfliers=False,meanline=False,medianprops=dict(linewidth=0))
    ax[i].set_xlabel('Importance for growth over '+label_list[i])
    combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
    del combined_data['subfindID']
    del combined_data['ID']
    del combined_data['BT_ratio']
    del combined_data['Unnamed: 0']
    for j in range(67):
        del combined_data['all_BT'+str(j)]
        del combined_data['lmbulge'+str(j)]
    #combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']>0.1].index)
    del combined_data['summed_merger_ratio']
    y=combined_data[outcome]
    del combined_data['d_BT_0.5']
    del combined_data['d_BT_1.0']
    del combined_data['d_BT_2.0']
    del combined_data['d_lmbulge_0.5']
    del combined_data['d_lmbulge_1.0']
    del combined_data['d_lmbulge_2.0']
    del combined_data['rel_BT_del_0.2']
    del combined_data['rel_BT_del_0.1']
    del combined_data['abs_BT_del_0.3']
    del combined_data['abs_BT_del_0.2']
    all_col=combined_data.columns
    X=combined_data
    model=model_list['Random_Forest']
    result=evaluate_model(X=X,y=y,model_name=model,splits=2,repeats=10)
    fitted_model=result[2]
    y_predict=predict_data(X_test=result[4],model=fitted_model)
    y_actual=result[5]
    X_test=result[4]
    y_train=result[6]
    score=result[0]
    score_sd=result[1]
    print(f'score is {score} with sd {score_sd} and MAE {result[3]}')
    y_train_predict=predict_data(X_test=result[7],model=fitted_model)
    model=model_list['Random_Forest']
    result=evaluate_model(X=X,y=y,model_name=model,splits=2,repeats=10)
    fitted_model=result[2]
    #print(f'R_2 is {result[0]}'+f' and MAE is {result[3]}')  
    r=permutation_importance(fitted_model,X_test,y_actual,n_repeats=30,random_state=11)
#spits out mean importance as first array and std importance as second. 
    all_imp_vals=[]
    all_cols=[]
    sorted_col=all_col[idx]
    for k in idx:
        #print(f'i is {i}')
        col_name=all_col[k]
        all_cols.append(col_name)
        #print(col_name)
        imp_values=r.importances[k,:]
        #print(imp_values)
        all_imp_vals.append(imp_values)
    print(all_imp_vals)    
    bplot=ax[i].boxplot(all_imp_vals,labels=all_cols,vert=0,showfliers=False,patch_artist=True,meanline=False,medianprops=dict(linewidth=0))
    ax[i].set_xlabel('Importance for growth over '+label_list[i])
    for patch in bplot['boxes']:
        patch.set_facecolor('lightblue')
plt.tight_layout()
plt.text(0.95,0.5,'Blue boxes for all SFGs, clear for low merger futures ',fontsize='small')
plt.show()
fig.savefig(plot_dir+'importance_plots_low_vs_all__mergers.png',dpi=1200,format='png')
'''
'''
#get pearson coefficients for each value and map in a heat map
outcome_list=['d_BT_0.5','d_BT_1.0','d_BT_2.0']
label_list=['0.5 Gyr','1 Gyr','2 Gyr']
nrows=1
ncols=2
fig,ax=plt.subplots(nrows=nrows,ncols=ncols,figsize=(10,5))
combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
del combined_data['subfindID']
del combined_data['ID']
del combined_data['BT_ratio']
del combined_data['Unnamed: 0']
del combined_data['d_BT_0.5']
del combined_data['d_BT_1.0']
del combined_data['d_BT_2.0']
del combined_data['d_lmbulge_0.5']
del combined_data['d_lmbulge_1.0']
del combined_data['d_lmbulge_2.0']
del combined_data['rel_BT_del_0.2']
del combined_data['rel_BT_del_0.1']
del combined_data['abs_BT_del_0.3']
del combined_data['abs_BT_del_0.2']
for j in range(67):
    del combined_data['all_BT'+str(j)]
    del combined_data['lmbulge'+str(j)]
combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']>0.1].index)
all_col=combined_data.columns
print(f'length of columns is {len(all_col)}')
X=combined_data
all_datasets=[X]
for data in all_datasets:
    for l,ncol in enumerate(data.columns):
       #print(ncol,data[ncol])
        median=np.nanmedian(data[ncol])
        print('median',median)
        data[ncol].fillna(median,inplace=True)
all_corr=np.array([])
for k,coly in enumerate(all_col):
    for m,colx in enumerate(all_col):
        correlation=pearsonr(X[coly],X[colx])[0]
        all_corr=np.append(all_corr,correlation)
print(f'length of all pearson is {all_corr.shape}')
#print(f'all correlations for {outcome_list[i]} is {all_corr}')
all_corr=np.reshape(all_corr,(len(all_col),len(all_col)))
#transpose teh triangle to get teh bottom half
mask=np.transpose(np.tri(len(all_col),k=-1))
#just get teh triangle by masking the array to make nans
all_corr=np.ma.array(all_corr,mask=mask)
print(all_corr.shape)
#print(f'all correlations for {outcome_list[i]} is {all_corr}')
cmap=cm.get_cmap('viridis')
cmap.set_bad('w')
im=ax[0].imshow(all_corr,cmap=cmap)
ax[0].set_xticks(np.arange(len(all_col)))
ax[0].set_xticklabels(all_col,fontsize='x-small',rotation='vertical')
ax[0].set_yticks(np.arange(len(all_col)))
ax[0].set_yticklabels(all_col,fontsize='x-small')
ax[0].set_title('Low future merger galaxies',y=1.05)
cbar=plt.colorbar(im,ax=ax[0],shrink=0.664)
#cbar.ax.set_yticklabels(fontsize='x-small',labels=['-1','0','1'])
#for n in range(len(all_col)):
#    for o in range(len(all_col)):
#        text=ax[i].text(o,n,all_corr[n,o],ha='center',va='center',color='w',fontsize='x-small')
plt.tight_layout()
#now do all gals
combined_data=pd.read_csv(my_dir+'all_combined_data_snap33')
del combined_data['subfindID']
del combined_data['ID']
del combined_data['BT_ratio']
del combined_data['Unnamed: 0']
del combined_data['d_BT_0.5']
del combined_data['d_BT_1.0']
del combined_data['d_BT_2.0']
del combined_data['d_lmbulge_0.5']
del combined_data['d_lmbulge_1.0']
del combined_data['d_lmbulge_2.0']
del combined_data['rel_BT_del_0.2']
del combined_data['rel_BT_del_0.1']
del combined_data['abs_BT_del_0.3']
del combined_data['abs_BT_del_0.2']
for j in range(67):
    del combined_data['all_BT'+str(j)]
    del combined_data['lmbulge'+str(j)]
combined_data=combined_data.drop(combined_data[combined_data['summed_merger_ratio']<0.1].index)
all_col=combined_data.columns
print(f'length of columns is {len(all_col)}')
X=combined_data
all_datasets=[X]
for data in all_datasets:
    for l,ncol in enumerate(data.columns):
       #print(ncol,data[ncol])
        median=np.nanmedian(data[ncol])
        print('median',median)
        data[ncol].fillna(median,inplace=True)
all_corr=np.array([])
for k,coly in enumerate(all_col):
    for m,colx in enumerate(all_col):
        correlation=pearsonr(X[coly],X[colx])[0]
        all_corr=np.append(all_corr,correlation)
print(f'length of all pearson is {all_corr.shape}')
#print(f'all correlations for {outcome_list[i]} is {all_corr}')
all_corr=np.reshape(all_corr,(len(all_col),len(all_col)))
#transpose teh triangle to get teh bottom half
mask=np.transpose(np.tri(len(all_col),k=-1))
#just get teh triangle by masking the array to make nans
all_corr=np.ma.array(all_corr,mask=mask)
print(all_corr.shape)
#print(f'all correlations for {outcome_list[i]} is {all_corr}')
cmap=cm.get_cmap('viridis')
cmap.set_bad('w')
im=ax[1].imshow(all_corr,cmap=cmap)
ax[1].set_xticks(np.arange(len(all_col)))
ax[1].set_xticklabels(all_col,fontsize='x-small',rotation='vertical')
ax[1].set_yticks(np.arange(len(all_col)))
ax[1].set_yticklabels(all_col,fontsize='x-small')
ax[1].set_title('High future merger galaxies',y=1.05)
cbar=plt.colorbar(im,ax=ax[1],shrink=0.664)
#cbar.ax.set_yticklabels(fontsize='x-small',labels=['-1','0','1'])
#for n in range(len(all_col)):
#    for o in range(len(all_col)):
#        text=ax[i].text(o,n,all_corr[n,o],ha='center',va='center',color='w',fontsize='x-small')
plt.tight_layout()        
fig.tight_layout()
plt.show()                    
fig.savefig(plot_dir+'pearson_heatmaps_small.png',dpi=1200,format='png')    
'''                                     

#use the new combined h5 files to look at random forest. Look at each snap individually and plot for each time interval.   

snap_list=[26,29,33,36,40,46,53]
delBT_t=['delbt05','delbt1','delbt2','delbt4']
label_list=['0.5 Gyr','1 Gyr','2 Gyr','4 Gyr']
fig,ax=plt.subplots(nrows=4,ncols=7,figsize=(14,8),sharey=True,sharex=True,gridspec_kw={'wspace':0,'hspace':0,'top':0.95,'bottom':0.05,'left':0.05,'right':0.95})
for i,snap in enumerate(snap_list):
    combined_data=hdf.read_hdf(my_dir+'combined_data_at_snap'+str(snap)+'.h5')
    y_quiet={}
    X_quiet={}
    y05=np.array([])
    y1=np.array([])
    y2=np.array([])
    y4=np.array([])
#print(combined_data33['summed_mass_ratio'])
#for key in combined_data33.keys():
#    print(key,combined_data33[key])
#get the merger Ids
    quiet_merg_idx=np.where(combined_data['summed_mass_ratio']<0.05)[0]
    #print(quiet_merg_idx)
#Y is outcome. create new dict of all quiet outcomes across all snaps
    y05=np.append(y05,combined_data['delBT'][:,0][quiet_merg_idx])
    y1=np.append(y1,combined_data['delBT'][:,1][quiet_merg_idx])
    y2=np.append(y2,combined_data['delBT'][:,2][quiet_merg_idx])
    y4=np.append(y4,combined_data['delBT'][:,3][quiet_merg_idx])
    y_quiet.update({'delbt05':y05,'delbt1':y1,'delbt2':y2,'delbt4':y4})
    quiet_Xdf=pd.DataFrame.from_dict(X_quiet)
    #keep as dict rather than converting to np array
    for key in combined_data.keys():
        if np.array(combined_data[key]).ndim==1:
            X_quiet.update({key:combined_data[key][quiet_merg_idx]})
        #print('length of x, new keys and combind ID',len(X),len(new_keys),len(combined_data33['ID']))
    print('X quiet is ',X_quiet,len(X_quiet))
    #print(X)
    #now convert into df
    quiet_Xdf=pd.DataFrame.from_dict(X_quiet)
    #print(new_keys)
    #print(quiet_Xdf.columns) 
    #get pearson coefficients for each value and map in a heat map
    #label_list=['0.5 Gyr','1 Gyr','2 Gyr','4 Gyr']
    del quiet_Xdf['subfindID']
    del quiet_Xdf['ID']
    del quiet_Xdf['BT_ratio']
    del quiet_Xdf['sim_name']
    del quiet_Xdf['snapNum']
    del quiet_Xdf['snap_fut']
    del quiet_Xdf['redshift']
    del quiet_Xdf['dir_plot']
    del quiet_Xdf['summed_mass_ratio']
    #del quiet_Xdf['lMgas']
    #del quiet_Xdf['lMcoldgas']
    #del quiet_Xdf['star_avg_vphiv']
    #del quiet_Xdf['star_avg_vRv']
    #del quiet_Xdf['star_avg_vzv']
    #del quiet_Xdf['gas_avg_vphiv']
    #del quiet_Xdf['gas_avg_vRv']
    #del quiet_Xdf['gas_avg_vzv']
    #del quiet_Xdf['star_avgRe_vphiv']
    #del quiet_Xdf['star_avgRe_vRv']
    #del quiet_Xdf['star_avgRe_vzv']
    #del quiet_Xdf['gas_avgRe_vphiv']
    #del quiet_Xdf['gas_avgRe_vRv']
    #del quiet_Xdf['gas_avgRe_vzv']
    #del quiet_Xdf['CAS_clump_idx']
    #del quiet_Xdf['Leaf_idx']
    #del quiet_Xdf['Rhalf']
    #del quiet_Xdf['dynbar']
#delete the following parapeters depending on interest in direction or gas fraction type
    #del quiet_Xdf['fgas']
    #del quiet_Xdf['fcoldgas']
    #del quiet_Xdf['frac_coldgas1kpc']
    #del quiet_Xdf['fcoldgas_inRe']
    #del quiet_Xdf['fcoldM_inRe']
    #del quiet_Xdf['gas_avg_v2zv2']
    #del quiet_Xdf['gas_avgRe_v2zv2']
    #del quiet_Xdf['gas_avg_v2Rv2']
    #del quiet_Xdf['gas_avgRe_v2Rv2']
    #del quiet_Xdf['gas_avg_v2phiv2']
    #del quiet_Xdf['gas_avgRe_v2phiv2']
    #del quiet_Xdf['star_avg_v2zv2']
    #del quiet_Xdf['star_avgRe_v2zv2']
    #del quiet_Xdf['star_avg_v2Rv2']
    #del quiet_Xdf['star_avgRe_v2Rv2']
    #del quiet_Xdf['star_avg_v2phiv2']
    #del quiet_Xdf['star_avgRe_v2phiv2']
    quiet_Xdf.replace('None',np.NaN)
    for j,t in enumerate(delBT_t):
        model=model_list['Random_Forest']
        result=evaluate_model(X=quiet_Xdf,y=y_quiet[t],model_name=model,splits=2,repeats=10)
        fitted_model=result[2]
        y_predict=predict_data(X_test=result[4],model=fitted_model)
        y_actual=result[5]
        X_test=result[4]
        y_train=result[6]
        score=result[0]
        score_sd=result[1]
        print(f'score is {score} with sd {score_sd} and MAE {result[3]}')
        y_train_predict=predict_data(X_test=result[7],model=fitted_model)
        correlation_test=(pearsonr(y_predict,y_actual)[0])**2
        correlation_train=(pearsonr(y_train_predict,y_train)[0])**2
        correlation_test=round(correlation_test,3)
        correlation_train=round(correlation_train,3)
        ax[j,i].scatter(y_actual,y_predict,c='black',s=5,label='Test values')
        #ax[j,i].scatter(y_train,y_train_predict,c='darkgray',s=4,label='Training values')
        #ax[j,i].legend(fontsize=10,labelspacing=0.2,columnspacing=1,loc='upper left')
        x=np.linspace(0,0.7,len(y_actual))
        y=x
        ax[j,i].set_xlim((0,0.7))
        ax[j,i].set_ylim((0,0.7))
        ax[j,i].plot(x,y,color='k',lw=0.5,ls='-')
        ax[j,i].text(0.3,0.6,r'Test $R^2=$'+str(correlation_test),fontsize=6)
        #ax[j,i].text(np.max(y_actual),np.max(y_predict),r'Training data $R^2=$'+str(correlation_train),fontsize=10,ha='right')
        ax[j,i].set_xlabel('Real BT growth',fontsize=8)
        ax[j,i].set_ylabel('Predicted BT growth over '+label_list[j],fontsize=8)
        ax[j,i].set_title('Snap '+str(snap_list[i]),fontsize=8)
plt.tight_layout()
plt.show()
fig.savefig(plot_dir+'all_model_plotsnew.png',dpi=1200,format='png')

'''
#Use all the snaps as a datasource and plot importance and predicted for quiet mergers only as defined by 0.05 cutoff. Plot for all time intervals.
snap_list=[26,29,33,36,40,46,53]
delBT_t=['delbt05','delbt1','delbt2','delbt4']
label_list=['0.5 Gyr','1 Gyr','2 Gyr','4 Gyr']
y_quiet={}
X_quiet={}
y05=np.array([])
y1=np.array([])
y2=np.array([])
y4=np.array([])
for snap in snap_list:
    combined_data=hdf.read_hdf(my_dir+'combined_data_at_snap'+str(snap)+'.h5')
#print(combined_data33['summed_mass_ratio'])
#for key in combined_data33.keys():
#    print(key,combined_data33[key])
#get the merger Ids
    quiet_merg_idx=np.where(combined_data['summed_mass_ratio']<0.05)[0]
    #print(quiet_merg_idx)
#Y is outcome. create new dict of all quiet outcomes across all snaps
    y05=np.append(y05,combined_data['delBT'][:,0][quiet_merg_idx])
    y1=np.append(y1,combined_data['delBT'][:,1][quiet_merg_idx])
    y2=np.append(y2,combined_data['delBT'][:,2][quiet_merg_idx])
    y4=np.append(y4,combined_data['delBT'][:,3][quiet_merg_idx])
#keep as dict rather than converting to np array. Again append values onto key for all snaps. 
combined_data=hdf.read_hdf(my_dir+'combined_data_at_snap33.h5')
for key in combined_data.keys():
    if np.array(combined_data[key]).ndim==1:
        all_X=np.array([])
        for snap in snap_list:
            combined_data=hdf.read_hdf(my_dir+'combined_data_at_snap'+str(snap)+'.h5')
            quiet_merg_idx=np.where(combined_data['summed_mass_ratio']<0.05)[0]
            all_X=np.append(all_X,combined_data[key][quiet_merg_idx])
            X_quiet.update({key:all_X})
print('snap',snap,'all len',len(combined_data['ID']),'quiet len',len(X_quiet['ID']))
print(X_quiet['ID'])
#now convert dict into df
y_quiet.update({'delbt05':y05,'delbt1':y1,'delbt2':y2,'delbt4':y4})
quiet_Xdf=pd.DataFrame.from_dict(X_quiet)
print('length of all vals is',len(quiet_Xdf['ID']))
del quiet_Xdf['subfindID']
del quiet_Xdf['ID']
del quiet_Xdf['BT_ratio']
del quiet_Xdf['sim_name']
del quiet_Xdf['snapNum']
del quiet_Xdf['snap_fut']
del quiet_Xdf['redshift']
del quiet_Xdf['dir_plot']
del quiet_Xdf['summed_mass_ratio']
#del quiet_Xdf['lMgas']
#del quiet_Xdf['lMcoldgas']
#del quiet_Xdf['star_avg_vphiv']
#del quiet_Xdf['star_avg_vRv']
#del quiet_Xdf['star_avg_vzv']
#del quiet_Xdf['gas_avg_vphiv']
#del quiet_Xdf['gas_avg_vRv']
#del quiet_Xdf['gas_avg_vzv']
#del quiet_Xdf['star_avgRe_vphiv']
#del quiet_Xdf['star_avgRe_vRv']
#del quiet_Xdf['star_avgRe_vzv']
#del quiet_Xdf['gas_avgRe_vphiv']
#del quiet_Xdf['gas_avgRe_vRv']
#del quiet_Xdf['gas_avgRe_vzv']
#del quiet_Xdf['CAS_clump_idx']
#del quiet_Xdf['Leaf_idx']
#del quiet_Xdf['Rhalf']
#del quiet_Xdf['dynbar']
#delete the following parapeters depending on interest in direction or gas fraction type
#del quiet_Xdf['fgas']
#del quiet_Xdf['fcoldgas']
#del quiet_Xdf['frac_coldgas1kpc']
#del quiet_Xdf['fcoldgas_inRe']
#del quiet_Xdf['fcoldM_inRe']
#del quiet_Xdf['gas_avg_v2zv2']
#del quiet_Xdf['gas_avgRe_v2zv2']
#del quiet_Xdf['gas_avg_v2Rv2']
#del quiet_Xdf['gas_avgRe_v2Rv2']
#del quiet_Xdf['gas_avg_v2phiv2']
#del quiet_Xdf['gas_avgRe_v2phiv2']
#del quiet_Xdf['star_avg_v2zv2']
#del quiet_Xdf['star_avgRe_v2zv2']
#del quiet_Xdf['star_avg_v2Rv2']
#del quiet_Xdf['star_avgRe_v2Rv2']
#del quiet_Xdf['star_avg_v2phiv2']
#del quiet_Xdf['star_avgRe_v2phiv2']
quiet_Xdf.replace('None',np.NaN)
model=model_list['Random_Forest']
#X=np.array(quiet_Xdf)
fig,ax=plt.subplots(nrows=2,ncols=4,figsize=(16,8))
for i,bt_t in enumerate(delBT_t):
    result=evaluate_model(X=quiet_Xdf,y=y_quiet[bt_t],model_name=model,splits=2,repeats=10)
    fitted_model=result[2]
    y_predict=predict_data(X_test=result[4],model=fitted_model)
    y_actual=result[5]
    X_test=result[4]
    y_train=result[6]
    score=result[0]
    score_sd=result[1]
    print(f'score is {score} with sd {score_sd} and MAE {result[3]}')
    y_train_predict=predict_data(X_test=result[7],model=fitted_model)
    correlation_test=(pearsonr(y_predict,y_actual)[0])**2
    correlation_train=(pearsonr(y_train_predict,y_train)[0])**2
    correlation_test=round(correlation_test,3)
    correlation_train=round(correlation_train,3)
    ax[0,i].scatter(y_actual,y_predict,c='black',s=4,label='Test values')
    ax[0,i].scatter(y_train,y_train_predict,c='darkgray',s=4,label='Training values')
    ax[0,i].legend(fontsize=10,labelspacing=0.2,columnspacing=1,loc='upper left')
    x=np.linspace(np.min(y_actual),np.max(y_actual),len(y_actual))
    y=x
    ax[0,i].set_xlim((np.min(y_actual),np.max(y_actual)))
    ax[0,i].set_ylim((np.min(y_predict),np.max(y_predict)))
    ax[0,i].plot(x,y,color='k',lw=0.5,ls='-')
    ax[0,i].text(np.min(y_actual),np.max(y_predict),r'Test $R^2=$'+str(correlation_test),fontsize=8)
    ax[0,i].text(np.max(y_actual),np.max(y_predict),r'Train $R^2=$'+str(correlation_train),fontsize=8,ha='right')
    ax[0,i].set_xlabel('Real BT growth over '+label_list[i],fontsize=8)
    ax[0,i].set_ylabel('Predicted BT growth over '+label_list[i],fontsize=8)
    #ax[0,i].set_title('All Quiet Merger Galaxies',fontsize=8)
    #print(f'R_2 is {result[0]}'+f' and MAE is {result[3]}')  
    r=permutation_importance(fitted_model,X_test,y_actual,n_repeats=30,random_state=11)
    #spits out mean importance as first array and std importance as second. 
    idx=np.argsort(r.importances_mean)
    #print(idx,len(idx))
    all_imp_vals=[]
    all_cols=[]
    for k in idx:
        #print(f'i is {i}')
        col_name=quiet_Xdf.columns[k]
        all_cols.append(col_name)
        #print(col_name)
        imp_values=r.importances[k,:]
        #print(imp_values)
        all_imp_vals.append(imp_values)
    #print(all_imp_vals)    
    ax[1,i].boxplot(all_imp_vals,vert=0,showfliers=False)
    ax[1,i].set_yticklabels(all_cols,fontsize=8)
    ax[1,i].set_xlabel('Permutation based importance for growth over '+label_list[i],fontsize=8)
    ax[1,i].set_title(r'Only $v^2phiRe/v^2$ considered',fontsize=8)
    plt.tight_layout()
#plt.tight_layout()
plt.show()
fig.savefig(plot_dir+'combined_model_and_importance.png',dpi=600,format='png')
'''    


'''
combined_data33=hdf.read_hdf(my_dir+'combined_data_at_snap33.h5')
#print(combined_data33['summed_mass_ratio'])
#for key in combined_data33.keys():
#    print(key,combined_data33[key])
#get the merger Ids
quiet_merg_idx=np.where(combined_data33['summed_mass_ratio']<0.01)[0]
high_merg_idx=np.where(combined_data33['summed_mass_ratio']>0.01)[0]
#print(quiet_merg_idx)
delBT_t=[0.5,1,2,4]
#Y is outcome 
y=combined_data33['delBT'][:,0]
quiet_y=y[quiet_merg_idx]
high_y=y[high_merg_idx]
#keep as dict rather than converting to np array
X_quiet={}
X_high={}
new_keys=[]
for key in combined_data33.keys():
    if np.array(combined_data33[key]).ndim==1:
        #print(key)
        #print(combined_data33[key])
        X_quiet.update({key:combined_data33[key][quiet_merg_idx]})
        X_high.update({key:combined_data33[key][high_merg_idx]})
        #print(key,X)
        new_keys.append(key)
#print('length of x, new keys and combind ID',len(X),len(new_keys),len(combined_data33['ID']))
print('X quiet is ',X_quiet,len(X_quiet))
#print(X)
#now convert into df
quiet_Xdf=pd.DataFrame.from_dict(X_quiet)
high_Xdf=pd.DataFrame.from_dict(X_high)
#print(new_keys)
#print(quiet_Xdf.columns) 
#get pearson coefficients for each value and map in a heat map
#label_list=['0.5 Gyr','1 Gyr','2 Gyr','4 Gyr']
del quiet_Xdf['subfindID']
del quiet_Xdf['ID']
del quiet_Xdf['BT_ratio']
del quiet_Xdf['sim_name']
del quiet_Xdf['snapNum']
del quiet_Xdf['snap_fut']
del quiet_Xdf['redshift']
del quiet_Xdf['dir_plot']
del quiet_Xdf['summed_mass_ratio']
del high_Xdf['subfindID']
del high_Xdf['ID']
del high_Xdf['BT_ratio']
del high_Xdf['sim_name']
del high_Xdf['snapNum']
del high_Xdf['snap_fut']
del high_Xdf['redshift']
del high_Xdf['dir_plot']
del high_Xdf['summed_mass_ratio']
#print(quiet_Xdf.columns)
#print(quiet_Xdf['SWbar'])
quiet_Xdf.replace('None',np.NaN)
high_Xdf.replace('None',np.NaN)
for ncol in quiet_Xdf.columns:
    median=np.nanmedian(quiet_Xdf[ncol])
    quiet_Xdf[ncol].fillna(median,inplace=True)
all_corr=np.array([])
for coly in quiet_Xdf.columns:
    for colx in quiet_Xdf.columns:
        correlation=pearsonr(quiet_Xdf[coly],quiet_Xdf[colx])[0]
        all_corr=np.append(all_corr,correlation)
print(f'length of all pearson is {all_corr.shape}')
#print(f'all correlations for {outcome_list[i]} is {all_corr}')
all_corr=np.reshape(all_corr,(len(quiet_Xdf.columns),len(quiet_Xdf.columns)))
#transpose teh triangle to get teh bottom half
mask=np.transpose(np.tri(len(quiet_Xdf.columns),k=-1))
#just get teh triangle by masking the array to make nans
all_corr=np.ma.array(all_corr,mask=mask)
print(all_corr.shape)
fig,ax=plt.subplots()
#print(f'all correlations for {outcome_list[i]} is {all_corr}')
cmap=cm.get_cmap('viridis')
cmap.set_bad('w')
im=ax.imshow(all_corr,cmap=cmap)
ax.set_xticks(np.arange(len(quiet_Xdf.columns)))
ax.set_xticklabels(quiet_Xdf.columns,fontsize='x-small',rotation='vertical')
ax.set_yticks(np.arange(len(quiet_Xdf.columns)))
ax.set_yticklabels(quiet_Xdf.columns,fontsize='x-small')
ax.set_title('Quiet future merger galaxies',y=1.05)
cbar=plt.colorbar(im,ax=ax,shrink=0.664)
#cbar.ax.set_yticklabels(fontsize='x-small',labels=['-1','0','1'])
#for n in range(len(all_col)):
#    for o in range(len(all_col)):
#        text=ax[i].text(o,n,all_corr[n,o],ha='center',va='center',color='w',fontsize='x-small')
plt.tight_layout()        
fig.tight_layout()
plt.show()                    
fig.savefig(plot_dir+'combined_pearson_heatmaps_quiet.png',dpi=1200,format='png')    
'''
'''
#plot individual parameters
combined_data=hdf.read_hdf(my_dir+'combined_data_at_snap33.h5')
print(combined_data.keys())
fig,ax=plt.subplots()
sns.regplot(combined_data['Total_m2'],combined_data['SWrbar'],color='dimgray',ax=ax,marker='o',scatter_kws={'s':3},line_kws={'lw':1})
ax.set_ylabel('SW bar')
ax.set_xlabel('My  Bar')
plt.show()
'''
'''
#use  snap 40 as the model ad test on all BT growth across snaps
snap_list=[26,29,33,36,40,46,53]
delBT_t=['delbt05','delbt1','delbt2','delbt4']
label_list=['0.5 Gyr','1 Gyr','2 Gyr','4 Gyr']
y_quiet={}
X_quiet={}
X_quiet40={}
y_quiet40={}
y05=np.array([])
y1=np.array([])
y2=np.array([])
y4=np.array([])
combined_data40=hdf.read_hdf(my_dir+'combined_data_at_snap40.h5')
quiet_merg_idx40=np.where(combined_data40['summed_mass_ratio']<0.05)[0]
y_quiet40.update({'delbt1':combined_data40['delBT'][:,1][quiet_merg_idx40]})
for key in combined_data40.keys():
    if np.array(combined_data40[key]).ndim==1:
        X_quiet40.update({key:combined_data40[key][quiet_merg_idx40]})
for snap in snap_list:
    combined_data=hdf.read_hdf(my_dir+'combined_data_at_snap'+str(snap)+'.h5')
#print(combined_data33['summed_mass_ratio'])
#for key in combined_data33.keys():
#    print(key,combined_data33[key])
#get the merger Ids
    quiet_merg_idx=np.where(combined_data['summed_mass_ratio']<0.05)[0]
    #print(quiet_merg_idx)
#Y is outcome. create new dict of all quiet outcomes across all snaps
    y05=np.append(y05,combined_data['delBT'][:,0][quiet_merg_idx])
    y1=np.append(y1,combined_data['delBT'][:,1][quiet_merg_idx])
    y2=np.append(y2,combined_data['delBT'][:,2][quiet_merg_idx])
    y4=np.append(y4,combined_data['delBT'][:,3][quiet_merg_idx])
#keep as dict rather than converting to np array. Again append values onto key for all snaps. 
combined_data=hdf.read_hdf(my_dir+'combined_data_at_snap33.h5')
for key in combined_data.keys():
    if np.array(combined_data[key]).ndim==1:
        all_X=np.array([])
        for snap in snap_list:
            combined_data=hdf.read_hdf(my_dir+'combined_data_at_snap'+str(snap)+'.h5')
            quiet_merg_idx=np.where(combined_data['summed_mass_ratio']<0.05)[0]
            all_X=np.append(all_X,combined_data[key][quiet_merg_idx])
            X_quiet.update({key:all_X})
print('snap',snap,'all len',len(combined_data['ID']),'quiet len',len(X_quiet['ID']))
print(X_quiet['ID'])
#now convert dict into df
y_quiet.update({'delbt05':y05,'delbt1':y1,'delbt2':y2,'delbt4':y4})
quiet_Xdf=pd.DataFrame.from_dict(X_quiet)
quiet_Xdf40=pd.DataFrame.from_dict(X_quiet40)
print('length of all vals is',len(quiet_Xdf['ID']))
del quiet_Xdf['subfindID']
del quiet_Xdf['ID']
del quiet_Xdf['BT_ratio']
del quiet_Xdf['sim_name']
del quiet_Xdf['snapNum']
del quiet_Xdf['snap_fut']
del quiet_Xdf['redshift']
del quiet_Xdf['dir_plot']
del quiet_Xdf['summed_mass_ratio']
del quiet_Xdf['lMgas']
del quiet_Xdf['lMcoldgas']
del quiet_Xdf['star_avg_vphiv']
del quiet_Xdf['star_avg_vRv']
del quiet_Xdf['star_avg_vzv']
del quiet_Xdf['gas_avg_vphiv']
del quiet_Xdf['gas_avg_vRv']
del quiet_Xdf['gas_avg_vzv']
del quiet_Xdf['star_avgRe_vphiv']
del quiet_Xdf['star_avgRe_vRv']
del quiet_Xdf['star_avgRe_vzv']
del quiet_Xdf['gas_avgRe_vphiv']
del quiet_Xdf['gas_avgRe_vRv']
del quiet_Xdf['gas_avgRe_vzv']
del quiet_Xdf['CAS_clump_idx']
del quiet_Xdf['Leaf_idx']
del quiet_Xdf['Rhalf']
del quiet_Xdf['dynbar']
#delete the following parapeters depending on interest in direction or gas fraction type
del quiet_Xdf['fgas']
del quiet_Xdf['fcoldgas']
#del quiet_Xdf['frac_coldgas1kpc']
#del quiet_Xdf['fcoldgas_inRe']
#del quiet_Xdf['fcoldM_inRe']
del quiet_Xdf['gas_avg_v2zv2']
del quiet_Xdf['gas_avgRe_v2zv2']
del quiet_Xdf['gas_avg_v2Rv2']
del quiet_Xdf['gas_avgRe_v2Rv2']
#del quiet_Xdf['gas_avg_v2phiv2']
del quiet_Xdf['gas_avgRe_v2phiv2']
del quiet_Xdf['star_avg_v2zv2']
del quiet_Xdf['star_avgRe_v2zv2']
del quiet_Xdf['star_avg_v2Rv2']
del quiet_Xdf['star_avgRe_v2Rv2']
#del quiet_Xdf['star_avg_v2phiv2']
del quiet_Xdf['star_avgRe_v2phiv2']
quiet_Xdf.replace('None',np.NaN)
del quiet_Xdf40['subfindID']
del quiet_Xdf40['ID']
del quiet_Xdf40['BT_ratio']
del quiet_Xdf40['sim_name']
del quiet_Xdf40['snapNum']
del quiet_Xdf40['snap_fut']
del quiet_Xdf40['redshift']
del quiet_Xdf40['dir_plot']
del quiet_Xdf40['summed_mass_ratio']
del quiet_Xdf40['lMgas']
del quiet_Xdf40['lMcoldgas']
del quiet_Xdf40['star_avg_vphiv']
del quiet_Xdf40['star_avg_vRv']
del quiet_Xdf40['star_avg_vzv']
del quiet_Xdf40['gas_avg_vphiv']
del quiet_Xdf40['gas_avg_vRv']
del quiet_Xdf40['gas_avg_vzv']
del quiet_Xdf40['star_avgRe_vphiv']
del quiet_Xdf40['star_avgRe_vRv']
del quiet_Xdf40['star_avgRe_vzv']
del quiet_Xdf40['gas_avgRe_vphiv']
del quiet_Xdf40['gas_avgRe_vRv']
del quiet_Xdf40['gas_avgRe_vzv']
del quiet_Xdf40['CAS_clump_idx']
del quiet_Xdf40['Leaf_idx']
del quiet_Xdf40['Rhalf']
del quiet_Xdf40['dynbar']
#delete the following parapeters depending on interest in direction or gas fraction type
del quiet_Xdf40['fgas']
del quiet_Xdf40['fcoldgas']
#del quiet_Xdf40['frac_coldgas1kpc']
#del quiet_Xdf40['fcoldgas_inRe']
#del quiet_Xdf40['fcoldM_inRe']
del quiet_Xdf40['gas_avg_v2zv2']
del quiet_Xdf40['gas_avgRe_v2zv2']
del quiet_Xdf40['gas_avg_v2Rv2']
del quiet_Xdf40['gas_avgRe_v2Rv2']
#del quiet_Xdf40['gas_avg_v2phiv2']
del quiet_Xdf40['gas_avgRe_v2phiv2']
del quiet_Xdf40['star_avg_v2zv2']
del quiet_Xdf40['star_avgRe_v2zv2']
del quiet_Xdf40['star_avg_v2Rv2']
del quiet_Xdf40['star_avgRe_v2Rv2']
#del quiet_Xdf40['star_avg_v2phiv2']
del quiet_Xdf40['star_avgRe_v2phiv2']
quiet_Xdf40.replace('None',np.NaN)
model=model_list['Random_Forest']
#X=np.array(quiet_Xdf)
all_datasets=[quiet_Xdf,quiet_Xdf40]
for data in all_datasets:
        for ncol in data.columns:
            #print(ncol,data[ncol])
            median=np.nanmedian(data[ncol])
            #print(ncol,median)
            data[ncol].fillna(median,inplace=True)
result=evaluate_model(X=quiet_Xdf40,y=y_quiet40['delbt1'],model_name=model,splits=2,repeats=10)
fitted_model=result[2]
y_predict=predict_data(X_test=quiet_Xdf,model=fitted_model)
y_actual=y_quiet['delbt1']
fig,ax=plt.subplots()
sns.regplot(y_actual,y_predict,ax=ax,color='blue',marker='o',scatter_kws={'s':3},line_kws={'lw':1})
y=np.linspace(0,0.7,len(y_predict))
x=y
ax.plot(x,y,color='k',lw=0.5)
plt.show()
'''
