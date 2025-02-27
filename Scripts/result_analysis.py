import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path
from statsmodels.stats.multitest import \
     multipletests as mult_test


file_path = file_path = os.path.dirname(os.path.realpath(__file__))
result_dir = str('../Results/Prediction_On_Images/Regression_Prediction/DNN_mutations/pan_drug_mutations/Double_network/CTRPv2')
#result_dir = str('../Results/Prediction_On_Images/Regression_Prediction/Mutation_interpolation_Oct_2024/pan_drug_pan_cancer/CCLE')
#result_dir = str('../Results/Prediction_On_Images/Regression_Prediction/LightGBM_ablation_Oct_2025/pan_drug_mutations')
result = pd.DataFrame(columns=['ge','cnv','mut_binary','mut_bin_del','mut_hubScore_inter5','mut_hubScore_inter4','mut_hubScore_inter3','mut_hubScore_inter2','mut_hubScore_inter1',
                               'mut_del_hubScore_inter5','mut_del_hubScore_inter4','mut_del_hubScore_inter3','mut_del_hubScore_inter2','mut_del_hubScore_inter1',
                               'mut_inter5_mean','mut_inter4_mean','mut_inter3_mean','mut_inter2_mean','mut_inter1_mean',
                               'mut_del_inter5_mean','mut_del_inter4_mean','mut_del_inter3_mean','mut_del_inter2_mean','mut_del_inter1_mean',
                               'R2_mean', 'R2_std','pCor_mean', 'pCor_std','sCor_mean', 'sCor_std'], index= range(len(os.listdir(result_dir))))
ind=0
num_partitions=20
data={'ge':'GE', 'cnv': 'CNV', 'mut_bin_del':'MUT',
      'ge_cnv':'GE+CNV', 'ge_mut_bin_del':'GE+MUT', 'mut_bin_del_cnv':'MUT+CNV'
      , 'Ge_mut_bin_del_cnv': 'GE+MUT+CNV',
      'mut_del_hubScore_inter3':'Mutation_del_Round3', 'mut_del_hubScore_inter2':'Mutation_del_Round2', 'mut_del_hubScore_inter1':'Mutation_del_Round1',
      'mut_hubScore_inter3':'Mutation_Round3', 'mut_hubScore_inter2':'Mutation_Round2', 'mut_hubScore_inter1':'Mutation_Round1',
      'ge_mut_del_hubScore_inter3':'GE+Mutation_del_Round3', 'ge_mut_del_hubScore_inter2':'GE+Mutation_del_Round2', 'ge_mut_del_hubScore_inter1':'GE+Mutation_del_Round1',
      'ge_mut_hubScore_inter3':'GE+Mutation_Round3', 'ge_mut_hubScore_inter2':'GE+Mutation_Round2', 'ge_mut_hubScore_inter1':'GE+Mutation_Round1'}
r2_all = {}
pcor_all={}
scor_all={}
print(os.listdir(result_dir))
for dir in os.listdir(result_dir):
    if dir=='ge_0.3333cnv_0.3333mut_hubScore_inter3_0.33340000000000003' or dir=='mut_bin_del_IGNORE'or dir=='CCLE' or dir=='GDSCv2' or dir=='mut_inter4_mean':
        continue
    if dir=='.DS_Store':
        continue
    if len(os.listdir(str(result_dir + '/'+ dir)))!=num_partitions or Path(str(result_dir + '/'+ dir+'/partition_19'+'/Prediction_Performance.txt')).is_file()==False:
        continue   

    data = []
    weights=[]
    for i in range(len(dir.split('_'))):
        if dir.split('_')[i]=='binary' or dir.split('_')[i]=='bin' or dir.split('_')[i]=='del'or dir.split('_')[i]=='hubScore'or dir.split('_')[i]=='inter1'or dir.split('_')[i]=='inter2'or dir.split('_')[i]=='inter3' or dir.split('_')[i]=='inter4' or dir.split('_')[i]=='inter5' or dir.split('_')[i]=='mean':
            continue
        if i==0:
            if dir.split('_')[i] == 'mut':
                if dir.split('_')[i+1] == 'hubScore':
                    data.append(str(dir.split('_')[i]+'_'+dir.split('_')[i+1]+'_'+dir.split('_')[i+2]))
                if dir.split('_')[i+1] == 'del':
                    data.append(str(dir.split('_')[i]+'_'+dir.split('_')[i+1]+'_'+dir.split('_')[i+2]+'_'+dir.split('_')[i+3]))
                if dir.split('_')[i+1] == 'binary':
                    data.append(str(dir.split('_')[i]+'_'+dir.split('_')[i+1]))
                if dir.split('_')[i+1] == 'bin':
                    data.append(str(dir.split('_')[i]+'_'+dir.split('_')[i+1]+'_'+dir.split('_')[i+2]))
                if dir.split('_')[i+1] == 'inter1' or dir.split('_')[i+1] == 'inter2' or dir.split('_')[i+1] == 'inter3' or dir.split('_')[i+1] == 'inter4' or dir.split('_')[i+1] == 'inter5':
                    data.append(str(dir.split('_')[i]+'_'+dir.split('_')[i+1]+'_'+dir.split('_')[i+2]))
            else:
                data.append(dir.split('_')[i])
        else:
            weights.append(float(dir.split('_')[i][:3]))
            if len(dir.split('_')[i])>3:
                if dir.split('_')[i][3:] == 'mut':
                    if dir.split('_')[i+1] != 'del':
                        data.append(str(dir.split('_')[i][3:]+'_'+dir.split('_')[i+1]+'_'+dir.split('_')[i+2]))
                    else:
                        data.append(str(dir.split('_')[i][3:]+'_'+dir.split('_')[i+1]+'_'+dir.split('_')[i+2]+'_'+dir.split('_')[i+3]))
                else:
                    data.append(dir.split('_')[i][3:])

    print(data)
    print(weights)
    r2_test_all = []
    pcor_test_all = []
    scor_test_all = []
    for partition in os.listdir(str(result_dir + '/'+ dir)):
        if partition=='.DS_Store':
            continue
        perf = pd.read_table(str(result_dir+'/'+dir+'/'+partition+'/Prediction_Performance.txt'))
        perf = perf.rename(columns={'Unnamed: 0': 'data_set'})
        perf.index = perf['data_set']
        perf.index.name = None
        r2_test_all.append(perf.loc['test']['R2'])
        pcor_test_all.append(perf.loc['test']['pCor'])
        scor_test_all.append(perf.loc['test']['sCor'])
    r2_all[dir] = r2_test_all
    r2_test_avg = np.mean(r2_test_all)
    r2_test_std = np.std(r2_test_all)

    pcor_all[dir] = pcor_test_all
    pcor_test_avg = np.mean(pcor_test_all)
    pcor_test_std = np.std(pcor_test_all)

    scor_all[dir] = scor_test_all
    scor_test_avg = np.mean(scor_test_all)
    scor_test_std = np.std(scor_test_all)
    for d in range(len(data)):
        result.loc[ind][data[d]] = weights[d]
    result.loc[ind]['R2_mean'] = r2_test_avg
    result.loc[ind]['R2_std'] = r2_test_std
    result.loc[ind]['pCor_mean'] = pcor_test_avg
    result.loc[ind]['pCor_std'] = pcor_test_std
    result.loc[ind]['sCor_mean'] = scor_test_avg
    result.loc[ind]['sCor_std'] = scor_test_std
    ind = ind+1
result=result.sort_values(by='R2_mean', ascending=False).reset_index(drop=True)
result.to_csv(os.path.join(result_dir, 'result.csv'))

###
### Pairwise t-test for mutation interpolation
stat = pd.DataFrame(columns=['Data', 'p_value_R2_mut_bin_del', 'R2_IsImprovement_mut_bin_del', 'p_value_pCor_mut_bin_del', 'pCor_IsImprovement_mut_bin_del', 'p_value_sCor_mut_bin_del', 'sCor_IsImprovement_mut_bin_del',
'p_value_R2_mut_bin', 'R2_IsImprovement_mut_bin', 'p_value_pCor_mut_bin', 'pCor_IsImprovement_mut_bin', 'p_value_sCor_mut_bin', 'sCor_IsImprovement_mut_bin'], index= range(len(r2_all.keys())))
ind=0
for key in r2_all.keys():
    stat.loc[ind]['Data'] = key
    stat.loc[ind]['p_value_R2_mut_bin_del']=stats.ttest_rel(r2_all[key], r2_all['mut_bin_del_1.0']).pvalue
    stat.loc[ind]['R2_IsImprovement_mut_bin_del'] = np.mean(r2_all[key])>np.mean(r2_all['mut_bin_del_1.0'])

    stat.loc[ind]['p_value_pCor_mut_bin_del']=stats.ttest_rel(pcor_all[key], pcor_all['mut_bin_del_1.0']).pvalue
    stat.loc[ind]['pCor_IsImprovement_mut_bin_del'] = np.mean(pcor_all[key])>np.mean(pcor_all['mut_bin_del_1.0'])

    stat.loc[ind]['p_value_sCor_mut_bin_del']=stats.ttest_rel(scor_all[key], scor_all['mut_bin_del_1.0']).pvalue
    stat.loc[ind]['sCor_IsImprovement_mut_bin_del'] = np.mean(scor_all[key])>np.mean(scor_all['mut_bin_del_1.0'])

    stat.loc[ind]['p_value_R2_mut_bin']=stats.ttest_rel(r2_all[key], r2_all['mut_binary_1.0']).pvalue
    stat.loc[ind]['R2_IsImprovement_mut_bin'] = np.mean(r2_all[key])>np.mean(r2_all['mut_binary_1.0'])

    stat.loc[ind]['p_value_pCor_mut_bin']=stats.ttest_rel(pcor_all[key], pcor_all['mut_binary_1.0']).pvalue
    stat.loc[ind]['pCor_IsImprovement_mut_bin'] = np.mean(pcor_all[key])>np.mean(pcor_all['mut_binary_1.0'])

    stat.loc[ind]['p_value_sCor_mut_bin']=stats.ttest_rel(scor_all[key], scor_all['mut_binary_1.0']).pvalue
    stat.loc[ind]['sCor_IsImprovement_mut_bin'] = np.mean(scor_all[key])>np.mean(scor_all['mut_binary_1.0'])

    ind=ind+1

stat = stat.dropna()
stat.to_csv(os.path.join(result_dir, 'stat_mut.csv'))


#Calculation of Performance improvement percentage (PIP):

r2_all['lgbm_iden'] = [0.5069228418898313,
 0.5119785058202777,
 0.5077786979033161,
 0.5031139400686966,
 0.5124243225084425,
 0.5099370409578671,
 0.511825110950799,
 0.5080833267384528,
 0.5177443192454134,
 0.5216724710259737,
 0.512137835714946,
 0.513834815847038,
 0.5226209966597383,
 0.5150528537596494,
 0.5081782725724884,
 0.5105741390061731,
 0.5120885760387763,
 0.5070858840358865,
 0.5042717799860053,
 0.5146595211079787,
 0.5072145586928938,
 0.5152508731181993,
 0.5111799233982766,
 0.5138374224069504,
 0.5074376227907456,
 0.5130091781788408,
 0.5113773046252047,
 0.5134570961752569,
 0.5152020218698827,
 0.5031699002192878,
 0.515504117949473,
 0.5153634068495385,
 0.5124943721784461,
 0.5159360998729218,
 0.5150767782620284,
 0.5096761706905777,
 0.5097710022761828,
 0.5098039873696915,
 0.5142135945945564,
 0.5113501430556485,
 0.5083497367344052,
 0.5094265479975213,
 0.5112700565210472,
 0.5138683418150334,
 0.5113108988864234,
 0.5084974566151366,
 0.511524751893037,
 0.5019502409882108,
 0.5095507386682527,
 0.5149712665806943]

r2_all_lgbm = r2_all['lgbm_iden'][:20]

comb = stat['Data'].values
baselines = ['mut_bin_del_1.0', 'mut_binary_1.0']
pip = pd.DataFrame(columns=baselines, index= range(len(comb)))
pip.index = comb

def sum_squares(l):
    squares = [x**2 for x in l] 
    return sum(squares)/len(squares)


for c in comb:
    for b in baselines:
        pip.loc[c][b] = (sum_squares(r2_all[c]) - sum_squares(r2_all[b]))/(sum_squares(r2_all[b]) - sum_squares(r2_all_lgbm)) *100
        
pip.to_csv(os.path.join(result_dir, 'pip.csv'))



""" #Pair-wise t-test
r2_all['lgbm_iden'] = r2_all['lgbm_iden'][:20]
stat = pd.DataFrame(columns=['Data', 'lgbm_i', 'Improve_lgbm_i', 'ge', 'Improve_ge', 'cnv', 'Improve_cnv', 'mut_bin_del', 'Improve_mut'], index= range(len(os.listdir(result_dir))))
ind=0
for key in r2_all.keys():
    stat.loc[ind]['Data'] = key
    stat.loc[ind]['lgbm_i']=stats.ttest_rel(r2_all[key], r2_all['lgbm_iden']).pvalue
    stat.loc[ind]['Improve_lgbm_i'] = np.mean(r2_all[key])>np.mean(r2_all['lgbm_iden'])

    stat.loc[ind]['ge']=stats.ttest_rel(r2_all[key], r2_all['ge_1']).pvalue
    stat.loc[ind]['Improve_ge'] = np.mean(r2_all[key])>np.mean(r2_all['ge_1'])

    stat.loc[ind]['cnv']=stats.ttest_rel(r2_all[key], r2_all['cnv_1']).pvalue
    stat.loc[ind]['Improve_cnv'] = np.mean(r2_all[key])>np.mean(r2_all['cnv_1'])

    stat.loc[ind]['mut_bin_del']=stats.ttest_rel(r2_all[key], r2_all['mut_bin_del_1']).pvalue
    stat.loc[ind]['Improve_mut'] = np.mean(r2_all[key])>np.mean(r2_all['mut_bin_del_1'])
    ind=ind+1

stat = stat.dropna()
stat.to_csv(os.path.join(result_dir, 'stat.csv'))


### Calculation of Performance improvement percentage (PIP):

r2_all_lgbm = r2_all['lgbm_iden']

comb = stat['Data'].values
baselines = ['cnv_1', 'mut_bin_del_1', 'ge_1', 'lgbm_iden']
pip = pd.DataFrame(columns=baselines, index= range(len(comb)))
pip.index = comb

def sum_squares(l):
    squares = [x**2 for x in l] 
    return sum(squares)/len(squares)


for c in comb:
    for b in baselines:
        pip.loc[c][b] = (sum_squares(r2_all[c]) - sum_squares(r2_all[b]))/(sum_squares(r2_all[b]) - sum_squares(r2_all_lgbm)) *100
        
pip.to_csv(os.path.join(result_dir, 'pip.csv')) """


""" 
# Generate all possible combinations of x, y, z such that x + y + z = 1
# Assuming x, y, z are integers in the range -10 to 10
split_ratios = []
x_all = [0,10,20,30,40,50,60,70,80,90,100]
for x in x_all:
    for y in x_all:
        z = 100 - x - y
        if 0 <= z <= 100:
            split_ratios.append((x/100, y/100, z/100))

feat = ['ge','mut_bin_del', 'cnv']

for split in split_ratios[:22]: 
    weight_list = list(split)
    feat = ['ge','mut_bin_del', 'cnv']
    ind = [i for i, e in enumerate(weight_list) if e != 0]
    feat = [feat[index] for index in ind]
    weight_list = [weight_list[index] for index in ind]  
    save_dir_name=''
    for i in range(len(feat)):
        save_dir_name = str(save_dir_name +feat[i]+'_'+str(weight_list[i]))
    print(save_dir_name) """


