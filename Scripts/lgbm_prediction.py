import pandas as pd
import os
from IGTD_Functions import min_max_transform, select_features_by_variation
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import r2_score
import json

def cancer_feat(feat = ['ge']):
    feat_data_dict = {'ge': 'cancer_gene_expression.tsv', 'cnv':'cancer_copy_number.tsv', 'mut_count':'cancer_mutation_count.tsv',
                    'cnv_disc':'cancer_discretized_copy_number.tsv', 'meth':'cancer_DNA_methylation.tsv','mut_bin_del':'Mutation_AID_binary_isDel.csv','mut_bin':'Mutation_AID_binary_isDel.csv',
                    'mut_del_inter3':'mut_del_corrected_ppi_3rdround.csv','mut_del_inter2':'mut_del_corrected_ppi_round_2.csv','mut_del_inter1':'mut_del_corrected_ppi_round_1.csv',
                    'mut_inter3':'mut_corrected_ppi_round_3.csv','mut_inter2':'mut_corrected_ppi_round_2.csv','mut_inter1':'mut_corrected_ppi_round_1.csv',
                    'mut_del_hubScore_inter3':'mut_corrected_ppi_del_hubScore_round_3.csv', 'mut_del_hubScore_inter2':'mut_corrected_ppi_del_hubScore_round_2.csv', 'mut_del_hubScore_inter1':'mut_corrected_ppi_del_hubScore_round_1.csv',
                    'mut_hubScore_inter3':'mut_corrected_ppi_hubScore_round_3.csv', 'mut_hubScore_inter2':'mut_corrected_ppi_hubScore_round_2.csv', 'mut_hubScore_inter1':'mut_corrected_ppi_hubScore_round_1.csv',
                    'mut_hubScore_inter4':'mut_corrected_ppi_hubScore_round_4.csv', 'mut_hubScore_inter5':'mut_corrected_ppi_hubScore_round_5.csv',
                    'mut_del_hubScore_inter4':'mut_corrected_ppi_del_hubScore_round_4.csv', 'mut_del_hubScore_inter5':'mut_corrected_ppi_del_hubScore_round_5.csv',
                    'mut_inter1_mean':'mut_ppi_combined_round_1_mean_all.csv','mut_inter2_mean':'mut_ppi_combined_round_2_mean_all.csv','mut_inter3_mean':'mut_ppi_combined_round_3_mean_all.csv',
                    'mut_inter4_mean':'mut_ppi_combined_round_4_mean_all.csv','mut_inter5_mean':'mut_ppi_combined_round_5_mean_all.csv',
                    'mut_del_inter1_mean':'mut_ppi_combined_round_1_mean_del.csv','mut_del_inter2_mean':'mut_ppi_combined_round_2_mean_del.csv','mut_del_inter3_mean':'mut_ppi_combined_round_3_mean_del.csv',
                    'mut_del_inter4_mean':'mut_ppi_combined_round_4_mean_del.csv','mut_del_inter5_mean':'mut_ppi_combined_round_5_mean_del.csv'}

    selected_genes=pd.read_table('../csa_data/raw_data/x_data/selected_genes.txt', header=None)
    gene_subset = list(selected_genes[0].values)
    gene_subset.insert(0, 'Cell-line')
    selected_cl=pd.read_table('../csa_data/raw_data/x_data/selected_cl.txt', header=None)
    common_cl = list(selected_cl[0].values)

    #feature_extraction
    data={}
    for f in feat:
        file = feat_data_dict[f]
        if f=='ge':
            ge= pd.read_table(str('../csa_data/raw_data/x_data/'+ file))
            ge.columns = ge.iloc[1] #Use only gene symbol column
            ge = ge.drop([0,1])
            ge = ge.rename(columns={ge.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=ge[gene_subset].set_index(['Cell-line'])
        if f=='cnv':
            cnv = pd.read_table(str('../csa_data/raw_data/x_data/'+file))
            cnv.columns = cnv.iloc[0]
            cnv = cnv.drop([0,1])
            cnv = cnv.rename(columns={cnv.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=cnv[gene_subset].set_index(['Cell-line'])
        if f=='mut_count':
            mut = pd.read_table(str('../csa_data/raw_data/x_data/'+file))
            mut.columns = mut.iloc[0]
            mut = mut.drop([0,1])
            mut = mut.rename(columns={mut.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut[gene_subset].set_index(['Cell-line'])
        if f=='mut_bin_del':  #### CHANGE feat_dist_method TO set #############
            mut_b = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            gene = mut_b['Gene_symbol']
            isdel = mut_b['isDeleterious']
            mut_b = mut_b.drop(mut_b.columns[list(range(0,16))], axis=1)
            mut_b.index=gene
            mut_b.index.name=None
            mut_b.columns= mut_b.iloc[0].values
            mut_b = mut_b.iloc[1:,:]
            ind_del = np.where(isdel==True)[0]-1
            mut_b=mut_b.iloc[ind_del]
            mut_b = mut_b.T
            mut_b = mut_b.loc[:,~mut_b.columns.duplicated()] # Remove duplicated columns
            mut_b = mut_b[~mut_b.index.duplicated(keep='first')] # Remove rows with duplicated cell-lines
            # We need to check for all zero columns
            mut_b = mut_b.loc[:, ~(mut_b == 0).all()]
            mut_b.insert(loc=0, column='Cell-line', value=mut_b.index.values)
            mut_b=mut_b.reset_index(drop=True)
            data[f]=mut_b[gene_subset].set_index(['Cell-line'])
        if f=='mut_bin':  #### CHANGE feat_dist_method TO set #############
            mut_b = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            gene = mut_b['Gene_symbol']
            mut_b = mut_b.drop(mut_b.columns[list(range(0,16))], axis=1)
            mut_b.index=gene
            mut_b.index.name=None
            mut_b.columns= mut_b.iloc[0].values
            mut_b = mut_b.iloc[1:,:]
            mut_b = mut_b.T
            mut_b = mut_b.loc[:,~mut_b.columns.duplicated()] # Remove duplicated columns
            mut_b = mut_b[~mut_b.index.duplicated(keep='first')] # Remove rows with duplicated cell-lines
            # We need to check for all zero columns
            mut_b = mut_b.loc[:, ~(mut_b == 0).all()]
            mut_b.insert(loc=0, column='Cell-line', value=mut_b.index.values)
            mut_b=mut_b.reset_index(drop=True)
            data[f]=mut_b[gene_subset].set_index(['Cell-line'])
        if f=='mut_del_inter3':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_del_inter2':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_del_inter1':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_inter3':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_inter2':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_inter1':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_hubScore_inter1':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_hubScore_inter2':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_hubScore_inter3':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_del_hubScore_inter1':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_del_hubScore_inter2':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_del_hubScore_inter3':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_hubScore_inter4':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_hubScore_inter5':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_del_hubScore_inter4':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_del_hubScore_inter5':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_inter1_mean' or f=='mut_inter2_mean' or f=='mut_inter3_mean' or f=='mut_inter4_mean' or f=='mut_inter5_mean':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        if f=='mut_del_inter1_mean' or f=='mut_del_inter2_mean' or f=='mut_del_inter3_mean' or f=='mut_del_inter4_mean' or f=='mut_del_inter5_mean':
            mut_inter = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
            mut_inter = mut_inter.rename(columns={mut_inter.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=mut_inter[gene_subset].set_index(['Cell-line'])
        
        data[f].index.name = None
        data[f]=data[f].loc[common_cl]
        data[f] = data[f].apply(pd.to_numeric)
        norm_data = min_max_transform(data[f].values)
        data[f] = pd.DataFrame(norm_data, columns=data[f].columns, index=data[f].index)
    return data

def drug_feat():
    drug = pd.read_table('../csa_data/raw_data/x_data/drug_mordred.tsv')
    drug = drug.set_index('improve_chem_id')
    drug.index.name = None
    return drug

    
def drug_selection(R2_filter=True, num_drugs=10):
    resp = pd.read_csv('../csa_data/raw_data/y_data/response.tsv', sep='\t', engine='c',
                      na_values=['na', '-', ''], header=0, index_col=None)
    res=resp[resp.source==study].reset_index(drop=True) # Choose from 1 study
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

def lgb_model(num_leaves, lgb_train, lgb_eval, lgb_params):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse', #metric(s) to be evaluated on the evaluation set(s)
        'num_leaves': num_leaves,   #max number of leaves in one tree
        'learning_rate': lgb_params['learning_rate'],
        #'feature_fraction': 0.9, #LightGBM will randomly select a subset of features on each iteration (tree)
        #'bagging_fraction': 0.8, #like feature_fraction, but this will randomly select part of data without resampling
        #'bagging_freq': 5, #frequency for bagging
        'verbose': -1,
        #'baggingSeed': 9,
        'random_state': lgb_params['random_state'],
	    'num_threads': 40
        #'device_type':'cuda'
    }
    evals_result = {}
    model = lgb.train(params,
                        lgb_train,
                        num_boost_round=lgb_params['num_boost_round'], #number of boosting iterations
                        valid_sets=lgb_eval,
                        callbacks=[lgb.early_stopping(stopping_rounds=lgb_params['stopping_rounds']),
                        lgb.record_evaluation(evals_result)])
    #_ = lgb.plot_metric(evals_result)
    return model

def train_lgb(X_train, X_val, Y_train, Y_val, num_leaves, seed, lgb_params):
    lgb_train = lgb.Dataset(X_train.drop(columns=['Cell_line']).astype(float), Y_train.astype(float))
    lgb_eval = lgb.Dataset(X_val.drop(columns=['Cell_line']).astype(float), Y_val.astype(float), reference=lgb_train)
    model = lgb_model(num_leaves, lgb_train, lgb_eval, lgb_params)
    return model

def train_lgb_pandrug(X_train, X_val, Y_train, Y_val, num_leaves, seed, lgb_params):
    lgb_train = lgb.Dataset(X_train.drop(columns=['Cell_line', 'Drug']).astype(float), Y_train.astype(float))
    lgb_eval = lgb.Dataset(X_val.drop(columns=['Cell_line','Drug']).astype(float), Y_val.astype(float), reference=lgb_train)
    model = lgb_model(num_leaves, lgb_train, lgb_eval, lgb_params)
    return model

def predict_hold_out(model, X_hold_out, Y_hold_out):
    pred = model.predict(X_hold_out.drop(columns=['Cell_line']).astype(float))
    rmse = np.sqrt(np.mean((pred - Y_hold_out.values)**2))
    r2 = r2_score(y_true=Y_hold_out.values, y_pred=pred)
    return rmse, r2

def predict_hold_out_pandrug(model, X_hold_out, Y_hold_out):
    pred = model.predict(X_hold_out.drop(columns=['Cell_line', 'Drug']).astype(float))
    rmse = np.sqrt(np.mean((pred - Y_hold_out.values)**2))
    r2 = r2_score(y_true=Y_hold_out.values, y_pred=pred)
    return rmse, r2

def multi_channel_prediction_drug_specific(drug, feat, savedir, response_dir):
    # Load the example drug response data, cell line gene expression image data, and drug descriptor image data
    res = pd.read_csv(response_dir, sep='\t', engine='c',
                      na_values=['na', '-', ''], header=0, index_col=None)
    res=res[res.source==study].reset_index(drop=True) # Choose from 1 study
    res=res[res.improve_chem_id==drug].reset_index(drop=True) # Choose drug
    #Duplicated cell-lines in the response data. - select the ones with higher r2 fit
    res = res.sort_values('r2fit', ascending=False).drop_duplicates('improve_sample_id', keep='first').reset_index(drop=False)
    #keep res for selected cell-lines
    selected_cl=pd.read_table('../csa_data/raw_data/x_data/selected_cl.txt', header=None)
    common_cl = list(selected_cl[0].values)
    res = res[res['improve_sample_id'].isin(common_cl)]
    num_sample = res.shape[0]
    seeds = list(range(0,num_data_partitions))
    canc_feat = cancer_feat(feat)
    keys = list(canc_feat.keys())
    feat_c = canc_feat[keys[0]]
    if len(keys)>1:
        for i in range(1, len(keys)):
            feat_c = feat_c.join(canc_feat[keys[i]], lsuffix=str('_'+keys[i-1]), rsuffix=str('_'+keys[i]), how='outer')
    #use only cell-lines with response data available
    feat_c = feat_c.loc[res['improve_sample_id'].values]

    #Add AUC value
    feat_c.insert(loc = 0, column = 'AUC', value = ['' for i in range(feat_c.shape[0])])
    for i in range(len(feat_c.index.values)):
        ind = np.where(res['improve_sample_id']==feat_c.index.values[i])[0]
        feat_c['AUC'][i] = res['auc'].iloc[ind].values[0] 
    feat_c.insert(loc = 0, column = 'Cell_line', value = feat_c.index.values)
    feat_c = feat_c.reset_index(drop=True)
    #model params
    lgb_params = {'learning_rate': 0.05, 'random_state':5, 'num_boost_round':500, 'stopping_rounds':30}
    r2_all=[]
    rmse_all=[]
    for seed in seeds:
        np.random.seed(seed)
        rand_sample_ID = np.random.permutation(num_sample)
        fold_size = int(num_sample / num_fold)
        sampleID = {}
        sampleID['trainID'] = rand_sample_ID[range(fold_size * (num_fold - 2))]
        sampleID['valID'] = rand_sample_ID[range(fold_size * (num_fold - 2), fold_size * (num_fold - 1))]
        sampleID['testID'] = rand_sample_ID[range(fold_size * (num_fold - 1), num_sample)]
        train = feat_c.iloc[sampleID['trainID']].reset_index(drop=True)
        val = feat_c.iloc[sampleID['valID']].reset_index(drop=True)
        test = feat_c.iloc[sampleID['testID']].reset_index(drop=True)

        # Run regression prediction
        X_train = train.drop(columns=['AUC'])
        Y_train = train['AUC']
        X_val = val.drop(columns=['AUC'])
        Y_val = val['AUC']
        X_test = test.drop(columns=['AUC'])
        Y_test = test['AUC']
        num_leaves=31
        model = train_lgb(X_train, X_val, Y_train, Y_val, num_leaves, seed, lgb_params)
        rmse, r2 = predict_hold_out(model, X_test, Y_test)
        r2_all.append(r2)
        rmse_all.append(rmse)

    output = lgb_params
    output['num_leaves'] = num_leaves
    output['seeds'] = seeds
    output['rmse'] = rmse_all
    output['r2'] = r2_all
    output['rmse_avg'] = np.mean(rmse_all)
    output['r2_avg'] = np.mean(r2_all)
    output['feat'] = feat
    output['drug'] = drug
    output['size_of_response'] = res.shape[0]
    output['study']= study
    # Create the directory for saving results
    result_dir = str('../Results/Prediction_On_Images/Regression_Prediction/'+ savedir+'/'+drug)
    os.makedirs(result_dir, exist_ok=True)
    savename='output.json'
    with open(result_dir + "/"+ savename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def multi_channel_prediction(feat, savedir, response_dir): # NEED EDITING FOR PAN_CANCER PAN DRUG
    # Load the example drug response data, cell line gene expression image data, and drug descriptor image data
    res = pd.read_csv(response_dir, sep='\t', engine='c',
                      na_values=['na', '-', ''], header=0, index_col=None)
    res=res[res.source==study].reset_index(drop=True) # Choose from 1 study
    #Duplicated cell-lines and drugs in the response data. - select the ones with higher r2 fit
    res = res.sort_values('r2fit', ascending=False).drop_duplicates(['improve_sample_id', 'improve_chem_id'], keep='first').reset_index(drop=False)
    #keep res for selected cell-lines
    selected_cl=pd.read_table('../csa_data/raw_data/x_data/selected_cl.txt', header=None)
    common_cl = list(selected_cl[0].values)
    res = res[res['improve_sample_id'].isin(common_cl)]
    num_sample = res.shape[0]
    seeds = list(range(0,num_data_partitions))
    canc_feat = cancer_feat(feat)
    keys = list(canc_feat.keys())
    feat_c = canc_feat[keys[0]]
    if len(keys)>1:
        for i in range(1, len(keys)):
            feat_c = feat_c.join(canc_feat[keys[i]], lsuffix=str('_'+keys[i-1]), rsuffix=str('_'+keys[i]), how='outer')
    #use only cell-lines with response data available
    feat_c = feat_c.loc[res['improve_sample_id'].values]
    #Drug features
    feat_d= drug_feat()
    feat_d = feat_d.loc[res['improve_chem_id'].values]

    #Concatenate cell-line and drug features
    feat_concat = pd.concat([feat_c.reset_index(drop=True), feat_d.reset_index(drop=True)], axis=1)

    # Create metadata DataFrame
    meta_data = res[['improve_sample_id', 'improve_chem_id', 'auc']]

    #Add AUC value
    feat_concat.insert(loc = 0, column = 'AUC', value = meta_data['auc'])
    feat_concat.insert(loc = 0, column = 'Cell_line', value = meta_data['improve_sample_id'])
    feat_concat.insert(loc = 0, column = 'Drug', value = meta_data['improve_chem_id'])

    #model params
    lgb_params = {'learning_rate': 0.05, 'random_state':5, 'num_boost_round':500, 'stopping_rounds':30}
    r2_all=[]
    rmse_all=[]
    for seed in seeds:
        np.random.seed(seed)
        rand_sample_ID = np.random.permutation(num_sample)
        fold_size = int(num_sample / num_fold)
        sampleID = {}
        sampleID['trainID'] = rand_sample_ID[range(fold_size * (num_fold - 2))]
        sampleID['valID'] = rand_sample_ID[range(fold_size * (num_fold - 2), fold_size * (num_fold - 1))]
        sampleID['testID'] = rand_sample_ID[range(fold_size * (num_fold - 1), num_sample)]
        train = feat_concat.iloc[sampleID['trainID']].reset_index(drop=True)
        val = feat_concat.iloc[sampleID['valID']].reset_index(drop=True)
        test = feat_concat.iloc[sampleID['testID']].reset_index(drop=True)

        # Run regression prediction
        X_train = train.drop(columns=['AUC'])
        Y_train = train['AUC']
        X_val = val.drop(columns=['AUC'])
        Y_val = val['AUC']
        X_test = test.drop(columns=['AUC'])
        Y_test = test['AUC']
        num_leaves=31
        model = train_lgb_pandrug(X_train, X_val, Y_train, Y_val, num_leaves, seed, lgb_params)
        rmse, r2 = predict_hold_out_pandrug(model, X_test, Y_test)
        r2_all.append(r2)
        rmse_all.append(rmse)

    output = lgb_params
    output['num_leaves'] = num_leaves
    output['seeds'] = seeds
    output['rmse'] = rmse_all
    output['r2'] = r2_all
    output['rmse_avg'] = np.mean(rmse_all)
    output['r2_avg'] = np.mean(r2_all)
    output['feat'] = feat
    output['size_of_response'] = res.shape[0]
    output['study']= study
    # Create the directory for saving results
    result_dir = str('../Results/Prediction_On_Images/Regression_Prediction/'+ savedir)
    os.makedirs(result_dir, exist_ok=True)
    savename='output.json'
    with open(result_dir + "/"+ savename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



num_data_partitions = 20  # Change this if needed
num_fold = 10
response_dir = '../csa_data/raw_data/y_data/response.tsv'
drug_specific = False
study = 'CTRPv2' #CTRPv2 or GDSCv2 or CCLE
#ge, cnv, mut_count, mut_bin_del, mut_del_inter3, mut_del_inter2, mut_del_inter1, mut_inter3, mut_inter2, mut_inter1
# mut_del_hubScore_inter3, mut_del_hubScore_inter2, mut_del_hubScore_inter1, mut_hubScore_inter3, mut_hubScore_inter2, mut_hubScore_inter1
# feats = [['ge'], ['cnv'], ['mut_del_hubScore_inter3'], ['mut_del_hubScore_inter2'], ['mut_del_hubScore_inter1'],
#          ['mut_hubScore_inter3'], ['mut_hubScore_inter2'], ['mut_hubScore_inter1'],
#          ['ge', 'cnv'], ['ge', 'mut_hubScore_inter3'], ['ge', 'mut_hubScore_inter2'], ['ge', 'mut_hubScore_inter1'],
#          ['ge', 'mut_del_hubScore_inter3'], ['ge', 'mut_del_hubScore_inter2'], ['ge', 'mut_del_hubScore_inter1']]

# feats = [['mut_del_hubScore_inter3'], ['mut_del_hubScore_inter2'], ['mut_del_hubScore_inter1'],
#          ['mut_hubScore_inter3'], ['mut_hubScore_inter2'], ['mut_hubScore_inter1'],
#          ['mut_del_hubScore_inter4'], ['mut_del_hubScore_inter5'], ['mut_hubScore_inter4'], ['mut_hubScore_inter5'],
#          ['mut_bin_del'], ['mut_bin']]

feats = [['mut_inter1_mean'], ['mut_inter2_mean'], ['mut_inter3_mean'], ['mut_inter4_mean'], ['mut_inter5_mean'],
         ['mut_del_inter1_mean'], ['mut_del_inter2_mean'], ['mut_del_inter3_mean'], ['mut_del_inter4_mean'], ['mut_del_inter5_mean']]

if drug_specific:
    drugs = drug_selection(R2_filter=True, num_drugs=10)
    for feat in feats:
        for drug in drugs:
            savename = '_'.join(feat)
            savedir = str('LightGBM_ablation_Oct_2025/drug_specific_mutations/'+study+'/'+savename)  ## CHANGE THIS
            multi_channel_prediction_drug_specific(drug, feat, savedir, response_dir)

else:
    for feat in feats:
        savename = '_'.join(feat)
        savedir = str('LightGBM_ablation_Oct_2025/pan_drug_mutations/'+study+'/'+savename)  ## CHANGE THIS
        multi_channel_prediction(feat, savedir, response_dir)



