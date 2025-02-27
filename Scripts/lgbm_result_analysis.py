import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import json
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import \
     multipletests as mult_test


file_path = file_path = os.path.dirname(os.path.realpath(__file__))
drug_specific=False

#Drug specific
if drug_specific:
    result_dir = str('../Results/Prediction_On_Images/Regression_Prediction/LightGBM_ablation_Oct_2025')
    dirs =os.listdir(result_dir)
    drugs = os.listdir(os.path.join(result_dir, dirs[0]))
    result_pd = pd.DataFrame(columns = dirs, index=drugs)
    for dir in dirs:
        for drug in os.listdir(os.path.join(result_dir, dir)):
            output = os.path.join(result_dir, dir, drug, 'output.json')
            with open(output) as f:
                d = json.load(f)
            result_pd[dir][drug] = d['r2_avg']
    print(result_pd)

    result_pd.to_csv('../Results/Prediction_On_Images/Regression_Prediction/LightGBM_ablation_Oct_2025/pan_drug_mutations/results.csv')

else:
    result_dir = str('../Results/Prediction_On_Images/Regression_Prediction/LightGBM_ablation_Oct_2025/pan_drug_mutations/CTRPv2')
    dirs =os.listdir(result_dir)
    result_pd = pd.DataFrame(columns = ['Data', 'R2_mean', 'R2_std'], index= range(len(dirs)))
    stat = pd.DataFrame(columns=['Data', 'p_value_R2_mut_bin_del', 'R2_IsImprovement_mut_bin_del',
        'p_value_R2_mut_bin', 'R2_IsImprovement_mut_bin'], index= range(len(dirs)))
    r2_all={}
    ind=0
    for dir in dirs:
        output = os.path.join(result_dir, dir, 'output.json')
        with open(output) as f:
            d = json.load(f)
        result_pd.loc[ind]['Data'] = dir
        result_pd.loc[ind]['R2_mean'] = np.mean(d['r2'])
        result_pd.loc[ind]['R2_std'] = np.std(d['r2'])
        r2_all[dir] = d['r2']
        ind=ind+1

    ind=0
    for key in dirs:
        stat.loc[ind]['Data'] = key
        stat.loc[ind]['p_value_R2_mut_bin_del']=stats.ttest_rel(r2_all[key], r2_all['mut_bin_del']).pvalue
        stat.loc[ind]['R2_IsImprovement_mut_bin_del'] = np.mean(r2_all[key])>np.mean(r2_all['mut_bin_del'])

        stat.loc[ind]['p_value_R2_mut_bin']=stats.ttest_rel(r2_all[key], r2_all['mut_bin']).pvalue
        stat.loc[ind]['R2_IsImprovement_mut_bin'] = np.mean(r2_all[key])>np.mean(r2_all['mut_bin'])
        ind=ind+1

    
    #Calculation of Performance improvement percentage (PIP):

    r2_lgbm_iden = [0.5069228418898313,
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

    r2_all_lgbm = r2_lgbm_iden[:20]

    comb = stat['Data'].values
    baselines = ['mut_bin_del', 'mut_bin']
    pip = pd.DataFrame(columns=baselines, index= range(len(comb)))
    pip.index = comb

    def sum_squares(l):
        squares = [x**2 for x in l] 
        return sum(squares)/len(squares)


    for c in comb:
        for b in baselines:
            pip.loc[c][b] = (sum_squares(r2_all[c]) - sum_squares(r2_all[b]))/(sum_squares(r2_all[b]) - sum_squares(r2_all_lgbm)) *100

    #Save
    stat.to_csv(os.path.join(result_dir, 'stat_mut.csv'))
    result_pd.to_csv(os.path.join(result_dir, 'result.csv'))
    pip.to_csv(os.path.join(result_dir, 'pip.csv'))

        