import numpy as np
import _pickle as cp
import shutil
import os
from pathlib import Path
import pandas as pd
import json


from Multi_channels_prediction_functions import get_model_parameter, load_data, get_data_for_cross_validation, \
    CNN2D_Regression_Analysis, CNN2D_Classification_Analysis, get_data_for_cross_validation_drug_spec



def multi_channel_prediction(channels, savedir, drug_dir, cancer_dir, response_dir, study):
    # Load the example drug response data, cell line gene expression image data, and drug descriptor image data
    res, ccl, drug = load_data(channels, drug_dir, cancer_dir, response_dir)
    res=res[res.source==study].reset_index(drop=True) # Choose from 1 study
    # Generate sample IDs for 10-fold cross-validation. 8 data folds for training, 1 data fold for validation,
    # and 1 data fold for testing
    num_sample = res.shape[0]
    seeds = list(range(0,num_data_partitions))
    for seed in seeds:
        np.random.seed(seed)
        rand_sample_ID = np.random.permutation(num_sample)
        fold_size = int(num_sample / num_fold)
        sampleID = {}
        sampleID['trainID'] = rand_sample_ID[range(fold_size * (num_fold - 2))]
        sampleID['valID'] = rand_sample_ID[range(fold_size * (num_fold - 2), fold_size * (num_fold - 1))]
        sampleID['testID'] = rand_sample_ID[range(fold_size * (num_fold - 1), num_sample)]
        # Run regression prediction
        # Create the directory for saving results
        result_dir = str('../Results/Prediction_On_Images/Regression_Prediction/'+ savedir +'/partition_'+str(seed))
        if os.path.exists(result_dir) and Path(str(result_dir+'/Prediction_Performance.txt')).is_file():
            continue
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)

        # Load network parameters
        para = get_model_parameter('../Data/Example_Model_Parameters/FCNN_Regressor.txt')
        subnetwork_para = get_model_parameter('../Data/Example_Model_Parameters/CNN2D_SubNetwork.txt')
        para.update(subnetwork_para)

        # Generate data for cross-validation analysis
        train, val, test = get_data_for_cross_validation(res, ccl, drug, sampleID)

        predResult, perM, perf, winningModel, batch_size = CNN2D_Regression_Analysis(train=train, resultFolder=result_dir,
                                                                                para=para, val=val, test=test)

        result = {}
        result['predResult'] = predResult       # Prediction values for the training, validation, and testing sets
        result['perM'] = perM                   # Loss values of training and validation during model training
        result['perf'] = perf                   # Prediction performance metrics
        result['winningModel'] = winningModel   # Model with the minimum validation loss
        result['batch_size'] = batch_size       # Batch size used in model training
        result['study'] = study
        # Save prediction performance and all data and results
        perf.to_csv(result_dir + '/Prediction_Performance.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
        output = open(result_dir + '/Result.pkl', 'wb')
        cp.dump(res, output)
        cp.dump(ccl, output)
        cp.dump(drug, output)
        cp.dump(result, output)
        cp.dump(sampleID, output)
        cp.dump(para, output)
        output.close()
def drug_selection(R2_filter=True, num_drugs=10):
    resp = pd.read_csv('../csa_data/raw_data/y_data/response.tsv', sep='\t', engine='c',
                      na_values=['na', '-', ''], header=0, index_col=None)
    res=resp[resp.source=='CTRPv2'].reset_index(drop=True) # Choose from 1 study
    if R2_filter:
        res = res[res['r2fit']>=0.8].reset_index(drop=True) # Filter based on R2 fit of dose response curve
    drug_id = res[['improve_chem_id']]
    drug_uniq = drug_id['improve_chem_id'].drop_duplicates()
    drug_names = pd.DataFrame({'improve_chem_id':drug_uniq.values, 'Response_size':['' for i in range(drug_uniq.shape[0])],
                                'Hits_size':['' for i in range(drug_uniq.shape[0])], 'Response_std_dev':['' for i in range(drug_uniq.shape[0])], 
                                'Response_mean':['' for i in range(drug_uniq.shape[0])], 'Hits/Response_size':['' for i in range(drug_uniq.shape[0])]})
    for i in range(len(drug_names)):
        ind = np.where(drug_id['improve_chem_id']==drug_names['improve_chem_id'].iloc[i])[0]
        # Check size of response matrix for each drug
        ind1 = np.where(res['improve_chem_id'] == drug_names['improve_chem_id'].iloc[i])[0]
        drug_names['Response_size'].iloc[i] = len(ind1)
        drug_names['Hits_size'].iloc[i] = sum(resp.iloc[ind]['auc'].values<0.5)
        drug_names['Response_std_dev'].iloc[i] = np.std(res.iloc[ind]['auc'])
        drug_names['Response_mean'].iloc[i] = np.mean(res.iloc[ind]['auc'])
        drug_names['Hits/Response_size'].iloc[i] = drug_names['Hits_size'].iloc[i]/drug_names['Response_size'].iloc[i]
    
    drug_names = drug_names[drug_names['Hits_size']>=20]
    drug_names = drug_names[drug_names['Hits/Response_size']<=0.7]
    drug_names = drug_names[drug_names['Response_size']>=500]
    drug_names_sort = drug_names.sort_values(by = ['Response_size'], ascending=False).reset_index(drop=True)
    drugs = list(drug_names_sort['improve_chem_id'])
    return drugs[:num_drugs]

def multi_channel_prediction_drug_specific(drug, channels, savedir, drug_dir, cancer_dir, response_dir):
    # Load the example drug response data, cell line gene expression image data, and drug descriptor image data
    res, ccl, drug2 = load_data(channels, drug_dir, cancer_dir, response_dir)
    res=res[res.source=='CTRPv2'].reset_index(drop=True) # Choose from 1 study
    res=res[res.improve_chem_id==drug].reset_index(drop=True) # Choose drug
    # Generate sample IDs for 10-fold cross-validation. 8 data folds for training, 1 data fold for validation,
    # and 1 data fold for testing
    num_sample = res.shape[0]
    seeds = list(range(0,num_data_partitions))
    for seed in seeds:
        np.random.seed(seed)
        rand_sample_ID = np.random.permutation(num_sample)
        fold_size = int(num_sample / num_fold)
        sampleID = {}
        sampleID['trainID'] = rand_sample_ID[range(fold_size * (num_fold - 2))]
        sampleID['valID'] = rand_sample_ID[range(fold_size * (num_fold - 2), fold_size * (num_fold - 1))]
        sampleID['testID'] = rand_sample_ID[range(fold_size * (num_fold - 1), num_sample)]
        # Run regression prediction
        # Create the directory for saving results
        result_dir = str('../Results/Prediction_On_Images/Regression_Prediction/'+ savedir +'/partition_'+str(seed))
        if os.path.exists(result_dir) and Path(str(result_dir+'/Prediction_Performance.txt')).is_file():
            continue
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)

        # Load network parameters
        para = get_model_parameter('../Data/Example_Model_Parameters/FCNN_Regressor.txt')
        subnetwork_para = get_model_parameter('../Data/Example_Model_Parameters/CNN2D_SubNetwork.txt')
        para.update(subnetwork_para)

        # Generate data for cross-validation analysis
        train, val, test = get_data_for_cross_validation_drug_spec(res, ccl, sampleID)

        predResult, perM, perf, winningModel, batch_size = CNN2D_Regression_Analysis(train=train, resultFolder=result_dir,
                                                                                para=para, val=val, test=test, batch_size=8)

        result = {}
        result['predResult'] = predResult       # Prediction values for the training, validation, and testing sets
        result['perM'] = perM                   # Loss values of training and validation during model training
        result['perf'] = perf                   # Prediction performance metrics
        result['winningModel'] = winningModel   # Model with the minimum validation loss
        result['batch_size'] = batch_size       # Batch size used in model training

        # Save prediction performance and all data and results
        perf.to_csv(result_dir + '/Prediction_Performance.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
        output = open(result_dir + '/Result.pkl', 'wb')
        cp.dump(res, output)
        cp.dump(ccl, output)
        cp.dump(drug, output)
        cp.dump(result, output)
        cp.dump(sampleID, output)
        cp.dump(para, output)
        output.close()

        #Save model parameters
        with open(str(result_dir +"/model_params.json"), "w") as f:
            json.dump(para, f, sort_keys=True, indent=4)

#### For comparison we also need analysis with just gene expression
# response_dir = '../csa_data/raw_data/y_data/response.tsv'
# drug_dir = '../Results/Table_To_Image_Conversion/Drug_40-40_50000Steps/data/'
# cancer_dir = str('../Results/Table_To_Image_Conversion/Cosmic_onco_50_50/3d_plot_images/'+'mut_bin_del'+'/data/')  ### CHANGE THIS
# savedir = str('Weights_3d_plot_results/Cosmic_onco_50_50/'+'mut_bin_del')  ## CHANGE THIS
# channels=1
drug_specific=False
num_data_partitions = 20  # Change this if needed
num_fold = 10
#multi_channel_prediction(channels=3, savedir='Ablation_study_30_partitions/Cosmic_onco_50_50/Ge_mut_bin_del_cnv') 
#multi_channel_prediction(channels=2, savedir='Ablation_study_30_partitions/Cosmic_onco_50_50/ge_cnv')
#multi_channel_prediction(channels=1, savedir='Ablation_study_30_partitions/Cosmic_onco_50_50/cnv')
#multi_channel_prediction(channels, savedir, drug_dir, cancer_dir, response_dir)
threeDplot=False
if threeDplot:
    if not drug_specific:
        data_dir = '../Results/Table_To_Image_Conversion/Mutation_interpolation_Oct_2024/3d_plot_images'
        count = 0
        for dir in os.listdir(data_dir):
            if count<40:
            #if count<35 or count>=40:
            #if count<30 or count>=35:
            #if count<25 or count>=30:
            #if count<20 or count>=25:
            #if count<15 or count>=20:
            #if count<10 or count>=15:
            #if count<5 or count>=10:
            #if count>=5:
            #if count<40 or count>=50:
            #if count<30 or count>=40:
            #if count<20 or count>=30:
            #if count<10 or count>=20:
            #if count>=10:
                count=count+1
                continue
            count=count+1
            
            savedir = str('Mutation_interpolation_Oct_2024/pan_drug_pan_cancer/'+dir)
            results_dir = str('../Results/Prediction_On_Images/Regression_Prediction/'+ savedir)
            if os.path.exists(results_dir):
                if len(os.listdir(results_dir))==num_data_partitions and Path(str(results_dir+'/partition_'+str(num_data_partitions-1)+'/Prediction_Performance.txt')).is_file():
                    continue   
                
            data = []
            weights=[]
            for i in range(len(dir.split('_'))):
                if dir.split('_')[i]=='bin' or dir.split('_')[i]=='del'or dir.split('_')[i]=='hubScore'or dir.split('_')[i]=='inter1'or dir.split('_')[i]=='inter2'or dir.split('_')[i]=='inter3'or dir.split('_')[i]=='inter4'or dir.split('_')[i]=='inter5':
                    continue
                if i==0:
                    if dir.split('_')[i] == 'mut':
                        if dir.split('_')[i+1] != 'del':
                            data.append(str(dir.split('_')[i]+'_'+dir.split('_')[i+1]+'_'+dir.split('_')[i+2]))
                        else:
                            data.append(str(dir.split('_')[i]+'_'+dir.split('_')[i+1]+'_'+dir.split('_')[i+2]+'_'+dir.split('_')[i+3]))
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
            channels = len(data)
            response_dir = '../csa_data/raw_data/y_data/response.tsv'
            drug_dir = '../Results/Table_To_Image_Conversion/Drug_40-40_50000Steps/data/'
            cancer_dir = str('../Results/Table_To_Image_Conversion/Mutation_interpolation_Oct_2024/3d_plot_images/'+dir+'/data/')     
            multi_channel_prediction(channels, savedir, drug_dir, cancer_dir, response_dir)

#dirs = ['mut_inter1_mean_1.0', 'mut_inter2_mean_1.0', 'mut_inter3_mean_1.0', 'mut_inter4_mean_1.0', 'mut_inter5_mean_1.0']
#dirs =  ['mut_del_inter1_mean_1.0', 'mut_del_inter2_mean_1.0', 'mut_del_inter3_mean_1.0', 'mut_del_inter4_mean_1.0', 'mut_del_inter5_mean_1.0']
#dirs= ['mut_bin_del_1.0', 'mut_bin_1.0', 'ge_1.0', 'cnv_1.0']
#dirs = ['mut_del_hubScore_inter4_1.0','mut_del_hubScore_inter5_1.0','mut_del_hubScore_inter3_1.0', 'mut_del_hubScore_inter2_1.0', 'mut_del_hubScore_inter1_1.0']
dirs=  ['mut_hubScore_inter3_1.0', 'mut_hubScore_inter2_1.0', 'mut_hubScore_inter1_1.0', 'mut_hubScore_inter4_1.0', 'mut_hubScore_inter5_1.0']
study='GDSCv2' #CCLE, CTRPv2, GDSCv2
for dir in dirs:
    response_dir = '../csa_data/raw_data/y_data/response.tsv'
    drug_dir = '../Results/Table_To_Image_Conversion/Drug_40-40_50000Steps/data/'
    cancer_dir = str('../Results/Table_To_Image_Conversion/Mutation_interpolation_Oct_2024/3d_plot_images/'+dir+'/data/')  
    savedir = str('Mutation_interpolation_Oct_2024/pan_drug_pan_cancer/'+study+'/'+dir)
    channels=1   
    multi_channel_prediction(channels, savedir, drug_dir, cancer_dir, response_dir,study)

### Drug specific
if drug_specific:
    drugs = drug_selection(R2_filter=True, num_drugs=10)
    data_dir = '../Results/Table_To_Image_Conversion/Mutation_interpolation_Oct_2024'
    #for drug in [drugs]:
        #drug = drugs[9] ### CHANGE THIS
        #count = 0
    drug = 'Drug_1493'
    #for dir in os.listdir(data_dir):
    dir='ge'
    #if count<50:
    #if count<40 or count>=50:
    #if count<30 or count>=40:
    #if count<20 or count>=30:
    #if count<10 or count>=20:
    #if count>=10:
        #count=count+1
        #continue
    #count=count+1
    
    #channels = len(data)
    channels=1
    response_dir = '../csa_data/raw_data/y_data/response.tsv'
    drug_dir = '../Results/Table_To_Image_Conversion/Drug_40-40_50000Steps/data/'
    cancer_dir = str('../Results/Table_To_Image_Conversion/Mutation_interpolation_Oct_2024/'+dir+'/data/')
    savedir = str('Mutation_interpolation_Oct_2024/drug_specific/HP_opt/'+dir+'/'+drug+'/hp4')
    results_dir = str('../Results/Prediction_On_Images/Regression_Prediction/'+ savedir)
    #if os.path.exists(results_dir):
    #    if len(os.listdir(results_dir))==num_data_partitions and Path(str(results_dir+'/partition_'+str(num_data_partitions-1)+'/Prediction_Performance.txt')).is_file():
    #        continue        
    multi_channel_prediction_drug_specific(drug, channels, savedir, drug_dir, cancer_dir, response_dir)


    """         
        data = []
        weights=[]
        for i in range(len(dir.split('_'))):
            if dir.split('_')[i]=='bin' or dir.split('_')[i]=='del':
                continue
            if i==0:
                if dir.split('_')[i] == 'mut':
                    data.append(str(dir.split('_')[i]+'_'+dir.split('_')[i+1]+'_'+dir.split('_')[i+2]))
                else:
                    data.append(dir.split('_')[i])
            else:
                weights.append(float(dir.split('_')[i][:3]))
                if len(dir.split('_')[i])>3:
                    if dir.split('_')[i][3:] == 'mut':
                        data.append(str(dir.split('_')[i][3:]+'_'+dir.split('_')[i+1]+'_'+dir.split('_')[i+2]))
                    else:
                        data.append(dir.split('_')[i][3:]) """


'''
# Run classification prediction
# Create the directory for saving results
result_dir = '../Results/Prediction_On_Images/Classification_Prediction'
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)

# Convert AUC values into response (AUC < 0.5) and non-response (AUC >= 0.5)
id_pos = np.where(res.AUC < 0.5)[0]
id_neg = np.setdiff1d(range(res.shape[0]), id_pos)
res.iloc[id_pos, 2] = 1
res.iloc[id_neg, 2] = 0
res.AUC = res.AUC.astype('int64')

# Load network parameters
para = get_model_parameter('../Data/Example_Model_Parameters/FCNN_Classifier.txt')
subnetwork_para = get_model_parameter('../Data/Example_Model_Parameters/CNN2D_SubNetwork.txt')
para.update(subnetwork_para)

# Generate data for cross-validation analysis
train, val, test = get_data_for_cross_validation(res, ccl, drug, sampleID)

predResult, perM, perf, winningModel, batch_size = CNN2D_Classification_Analysis(train=train, num_class=2,
    resultFolder=result_dir, class_weight='balanced', para=para, val=val, test=test)

result = {}
result['predResult'] = predResult       # Prediction values for the training, validation, and testing sets
result['perM'] = perM                   # Loss values of training and validation during model training
result['perf'] = perf                   # Prediction performance metrics
result['winningModel'] = winningModel   # Model with the minimum validation loss
result['batch_size'] = batch_size       # Batch size used in model training

# Save prediction performance and all data and results
perf.to_csv(result_dir + '/Prediction_Performance.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
output = open(result_dir + '/Result.pkl', 'wb')
cp.dump(res, output)
cp.dump(ccl, output)
cp.dump(drug, output)
cp.dump(result, output)
cp.dump(sampleID, output)
cp.dump(para, output)
output.close()
'''
