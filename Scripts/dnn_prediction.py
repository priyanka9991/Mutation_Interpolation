import pandas as pd
import os
from IGTD_Functions import min_max_transform, select_features_by_variation
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import r2_score
from scipy import stats
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import logging
import sys
import shutil 
from sklearn.preprocessing import StandardScaler

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(f'DNN model training')

def calculate_batch_size(num_sample, paraDNN):
    # max_half_num_batch: the number of batches will not be larger than 2 * max_half_num_batch
    max_half_num_batch = paraDNN['max_half_num_batch']
    if num_sample < max_half_num_batch * 4:
        batch_size = 2
    elif num_sample < max_half_num_batch * 8:
        batch_size = 4
    elif num_sample < max_half_num_batch * 16:
        batch_size = 8
    elif num_sample < max_half_num_batch * 32:
        batch_size = 16
    elif num_sample < max_half_num_batch * 64:
        batch_size = 32
    elif num_sample < max_half_num_batch * 128:
        batch_size = 64
    elif num_sample < max_half_num_batch * 256:
        batch_size = 128
    else:
        batch_size = 256

    return batch_size

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

def dnn_model_double_network(X_train_cancer, X_val_cancer, X_train_drug, X_val_drug, Y_train, Y_val, model_params):
    logger.info('****** START NEW MODEL TRAINING *****')
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert data to tensors and move to device
    X_train_cancer_tensor = torch.FloatTensor(X_train_cancer.values).to(device)
    X_train_drug_tensor = torch.FloatTensor(X_train_drug.values).to(device)
    Y_train_tensor = torch.FloatTensor(Y_train.values).to(device)
    X_val_cancer_tensor = torch.FloatTensor(X_val_cancer.values).to(device)
    X_val_drug_tensor = torch.FloatTensor(X_val_drug.values).to(device)
    Y_val_tensor = torch.FloatTensor(Y_val.values).to(device)

    # Create datasets and dataloaders
    
    train_dataset = TensorDataset(X_train_cancer_tensor, X_train_drug_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True)
    val_dataset = TensorDataset(X_val_cancer_tensor, X_val_drug_tensor, Y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'])

    # Define model architecture
    cancer_input_size = X_train_cancer.shape[1]
    drug_input_size = X_train_drug.shape[1]

    seed = torch.manual_seed(model_params['seed'])
    lr = model_params['lr']
    dropout = model_params['dropout']
    h_sizes = model_params['h_sizes']
    n_epochs=model_params['n_epochs']

    # Cancer subnetwork
    cancer_net = nn.Sequential()
    input_size = cancer_input_size
    for k in range(len(h_sizes)):
        cancer_net.add_module(str('cancer_hidden_'+str(k)), nn.Linear(input_size, h_sizes[k]))
        cancer_net.add_module(str('cancer_bn_'+str(k)), nn.BatchNorm1d(h_sizes[k]))
        cancer_net.add_module(str('cancer_dropout_'+str(k)), nn.Dropout(dropout))
        cancer_net.add_module(str('cancer_relu_'+str(k)), nn.ReLU())
        input_size = h_sizes[k]

    # Drug subnetwork  
    drug_net = nn.Sequential()
    input_size = drug_input_size
    for k in range(len(h_sizes)):
        drug_net.add_module(str('drug_hidden_'+str(k)), nn.Linear(input_size, h_sizes[k]))
        drug_net.add_module(str('drug_bn_'+str(k)), nn.BatchNorm1d(h_sizes[k]))
        drug_net.add_module(str('drug_dropout_'+str(k)), nn.Dropout(dropout))
        drug_net.add_module(str('drug_relu_'+str(k)), nn.ReLU())
        input_size = h_sizes[k]

    # Combined network
    if model_params['sigmoid_activation']:
        combined_net = nn.Sequential(
            nn.Linear(h_sizes[-1]*2, h_sizes[-1]),
            nn.BatchNorm1d(h_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_sizes[-1], 1),
            nn.Sigmoid()
        )
    else: 
        combined_net = nn.Sequential(
            nn.Linear(h_sizes[-1]*2, h_sizes[-1]),
            nn.BatchNorm1d(h_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_sizes[-1], 1)
        )

    # Create complete model
    class CompleteModel(nn.Module):
        def __init__(self, cancer_net, drug_net, combined_net):
            super().__init__()
            self.cancer_net = cancer_net
            self.drug_net = drug_net
            self.combined_net = combined_net

        def forward(self, x_cancer, x_drug):
            cancer_out = self.cancer_net(x_cancer)
            drug_out = self.drug_net(x_drug)
            combined = torch.cat((cancer_out, drug_out), dim=1)
            return self.combined_net(combined)

    model = CompleteModel(cancer_net, drug_net, combined_net).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_loss = float('inf')
    patience = model_params['patience']
    counter = 0
    best_epoch = 0

    for epoch in range(model_params['n_epochs']):
        model.train()
        train_loss = 0
        for cancer_batch, drug_batch, y_batch in train_loader:
            cancer_batch, drug_batch, y_batch = cancer_batch.to(device), drug_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(cancer_batch, drug_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_predictions=[]
        val_actuals=[]
        with torch.no_grad():
            for cancer_batch, drug_batch, y_batch in val_loader:
                cancer_batch, drug_batch, y_batch = cancer_batch.to(device), drug_batch.to(device), y_batch.to(device)
                y_pred = model(cancer_batch, drug_batch)
                val_predictions.extend(y_pred.squeeze().cpu().tolist())
                val_actuals.extend(y_batch.cpu().tolist())
                val_loss += criterion(y_pred, y_batch.unsqueeze(1)).item()
        # Print training progress
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'Epoch [{epoch+1}/{n_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        val_predictions = np.array(val_predictions)
        val_actuals = np.array(val_actuals)

        # Training performance
        model.eval()
        train_predictions = []
        train_actuals = []
        with torch.no_grad():
            for cancer_batch, drug_batch, y_batch in train_loader:
                # Move batch to device
                cancer_batch, drug_batch, y_batch = cancer_batch.to(device), drug_batch.to(device), y_batch.to(device)
                y_pred = model(cancer_batch, drug_batch)
                train_predictions.extend(y_pred.squeeze().cpu().tolist())
                train_actuals.extend(y_batch.cpu().tolist())
        train_predictions = np.array(train_predictions)
        train_actuals = np.array(train_actuals)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Calculate metrics - validation
            r2_val = r2_score(val_actuals, val_predictions)
            rmse_val = np.sqrt(np.mean((val_actuals - val_predictions) ** 2)) 
            pCor_val, pearson_p = stats.pearsonr(val_actuals, val_predictions)
            sCor_val, spearman_p = stats.spearmanr(val_actuals, val_predictions)
            # Calculate metrics - train
            r2_train = r2_score(train_actuals, train_predictions)
            rmse_train = np.sqrt(np.mean((train_actuals - train_predictions) ** 2)) 
            pCor_train, pearson_p = stats.pearsonr(train_actuals, train_predictions)
            sCor_train, spearman_p = stats.spearmanr(train_actuals, train_predictions)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info('..... EARLY STOPPING ......')
                break
    val_perf = {}
    val_perf['r2'] = r2_val
    val_perf['rmse'] = rmse_val
    val_perf['pCor'] = pCor_val
    val_perf['sCor'] = sCor_val
    val_perf['best_epoch'] = best_epoch

    train_perf = {}
    train_perf['r2'] = r2_train
    train_perf['rmse'] = rmse_train
    train_perf['pCor'] = pCor_train
    train_perf['sCor'] = sCor_train
    train_perf['best_epoch'] = best_epoch
    return model, val_perf, train_perf



def dnn_model(X_train, X_val, Y_train, Y_val, model_params):
    logger.info('****** START NEW MODEL TRAINING *****')
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert data to tensors and move to device
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    Y_train_tensor = torch.FloatTensor(Y_train.values).to(device)
    X_val_tensor = torch.FloatTensor(X_val.values).to(device)
    Y_val_tensor = torch.FloatTensor(Y_val.values).to(device)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'])

    # Define model architecture
    input_size = X_train.shape[1]

    seed = torch.manual_seed(model_params['seed'])
    lr = model_params['lr']
    net = nn.Sequential()
    dropout = model_params['dropout']
    h_sizes = model_params['h_sizes']
    for k in range(len(h_sizes)):
        net.add_module(str('hidden_'+str(k)),nn.Linear(input_size, h_sizes[k]))
        net.add_module(str('bn_'+str(k)), nn.BatchNorm1d(h_sizes[k]))
        net.add_module(str('dropout_'+str(k)),nn.Dropout(dropout))
        net.add_module(str('relu_'+str(k)),nn.ReLU())
        input_size = h_sizes[k]
    net.add_module('output',nn.Linear(h_sizes[-1], 1))
    if model_params['sigmoid_activation']:
        net.add_module('sigmoid',nn.Sigmoid())
    model = net.to(device)
    
    # model = nn.Sequential(
    #     nn.Linear(input_size, 128),
    #     nn.ReLU(),
    #     nn.Dropout(0.2),
    #     nn.Linear(128, 64),
    #     nn.ReLU(), 
    #     nn.Dropout(0.2),
    #     nn.Linear(64, 1)
    # ).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    n_epochs = model_params['n_epochs']
    best_val_loss = float('inf')
    best_epoch=0
    patience = model_params['patience']
    counter = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            # Move batch to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_predictions = []
        val_actuals = []
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Move batch to device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                y_pred = model(X_batch)
                val_predictions.extend(y_pred.squeeze().cpu().tolist())
                val_actuals.extend(y_batch.cpu().tolist())
                val_loss += criterion(y_pred, y_batch.unsqueeze(1)).item()
        # Print training progress
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'Epoch [{epoch+1}/{n_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        val_predictions = np.array(val_predictions)
        val_actuals = np.array(val_actuals)

        # Training performance
        model.eval()
        train_predictions = []
        train_actuals = []
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                # Move batch to device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                train_predictions.extend(y_pred.squeeze().cpu().tolist())
                train_actuals.extend(y_batch.cpu().tolist())
        train_predictions = np.array(train_predictions)
        train_actuals = np.array(train_actuals)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Calculate metrics - validation
            r2_val = r2_score(val_actuals, val_predictions)
            rmse_val = np.sqrt(np.mean((val_actuals - val_predictions) ** 2)) 
            pCor_val, pearson_p = stats.pearsonr(val_actuals, val_predictions)
            sCor_val, spearman_p = stats.spearmanr(val_actuals, val_predictions)
            # Calculate metrics - train
            r2_train = r2_score(train_actuals, train_predictions)
            rmse_train = np.sqrt(np.mean((train_actuals - train_predictions) ** 2)) 
            pCor_train, pearson_p = stats.pearsonr(train_actuals, train_predictions)
            sCor_train, spearman_p = stats.spearmanr(train_actuals, train_predictions)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info('..... EARLY STOPPING ......')
                break
    val_perf = {}
    val_perf['r2'] = r2_val
    val_perf['rmse'] = rmse_val
    val_perf['pCor'] = pCor_val
    val_perf['sCor'] = sCor_val
    val_perf['best_epoch'] = best_epoch

    train_perf = {}
    train_perf['r2'] = r2_train
    train_perf['rmse'] = rmse_train
    train_perf['pCor'] = pCor_train
    train_perf['sCor'] = sCor_train
    train_perf['best_epoch'] = best_epoch
    return model, val_perf, train_perf

def predict_hold_out(model, X_hold_out, Y_hold_out, model_params):
    logger.info('****** START PREDICTIONS *****')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert test data to tensors and move to GPU
    X_hold_out_tensor = torch.FloatTensor(X_hold_out.values).to(device)
    Y_hold_out_tensor = torch.FloatTensor(Y_hold_out.values).to(device)

    # Create test dataloader
    test_dataset = TensorDataset(X_hold_out_tensor, Y_hold_out_tensor)
    test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'])

    # Get predictions
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Move batch to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            predictions.extend(y_pred.squeeze().cpu().tolist())
            actuals.extend(y_batch.cpu().tolist())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    r2 = r2_score(actuals, predictions)
    # Calculate Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(actuals, predictions)
    # Calculate Spearman correlation 
    spearman_corr, spearman_p = stats.spearmanr(actuals, predictions)
    logger.info(f'Test Pearson correlation is {pearson_corr:.3f} (p={pearson_p:.3e})')
    logger.info(f'Test Spearman correlation is {spearman_corr:.3f} (p={spearman_p:.3e})')
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    logger.info (f'Test R2 is {r2} and test rmse is {rmse}')
    test_perf = {}
    test_perf['r2'] = r2
    test_perf['rmse'] = rmse
    test_perf['pCor'] = pearson_corr
    test_perf['sCor'] = spearman_corr
    return test_perf

def predict_hold_out_double_network(model, X_test_cancer, X_test_drug, Y_test, model_params):
    logger.info('****** START PREDICTIONS *****')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert test data to tensors and move to GPU
    X_test_cancer_tensor = torch.FloatTensor(X_test_cancer.values).to(device)
    X_test_drug_tensor = torch.FloatTensor(X_test_drug.values).to(device) 
    Y_test_tensor = torch.FloatTensor(Y_test.values).to(device)

    # Create test dataloader
    test_dataset = TensorDataset(X_test_cancer_tensor, X_test_drug_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'])

    # Get predictions
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for cancer_batch, drug_batch, y_batch in test_loader:
            # Move batch to device
            cancer_batch, drug_batch, y_batch = cancer_batch.to(device), drug_batch.to(device), y_batch.to(device)
            
            y_pred = model(cancer_batch, drug_batch)
            predictions.extend(y_pred.squeeze().cpu().tolist())
            actuals.extend(y_batch.cpu().tolist())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    r2 = r2_score(actuals, predictions)
    # Calculate Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(actuals, predictions)
    # Calculate Spearman correlation 
    spearman_corr, spearman_p = stats.spearmanr(actuals, predictions)
    logger.info(f'Test Pearson correlation is {pearson_corr:.3f} (p={pearson_p:.3e})')
    logger.info(f'Test Spearman correlation is {spearman_corr:.3f} (p={spearman_p:.3e})')
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    logger.info(f'Test R2 is {r2} and test rmse is {rmse}')
    test_perf = {}
    test_perf['r2'] = r2
    test_perf['rmse'] = rmse
    test_perf['pCor'] = pearson_corr
    test_perf['sCor'] = spearman_corr
    return test_perf
    


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
    val_r2_all=[]
    val_rmse_all=[]
    best_epoch_all =[]
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
        model, val_perf = dnn_model(X_train, X_val, Y_train, Y_val)
        test_perf = predict_hold_out(model, X_test, Y_test)
        r2_all.append(test_perf['r2'])
        rmse_all.append(test_perf['rmse'])
        val_r2_all.append(val_perf['r2'])
        val_rmse_all.append(val_perf['rmse'])
        best_epoch_all.append(val_perf['best_e[poch]'])

    output = lgb_params
    output['seeds'] = seeds
    output['best_epochs']=best_epoch_all
    output['test_rmse'] = rmse_all
    output['test_r2'] = r2_all
    output['test_rmse_avg'] = np.mean(rmse_all)
    output['test_r2_avg'] = np.mean(r2_all)
    output['val_rmse'] = val_rmse_all
    output['val_r2'] = val_r2_all
    output['val_rmse_avg'] = np.mean(val_rmse_all)
    output['val_r2_avg'] = np.mean(val_r2_all)
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

def data_single_network(response_dir):
    # Load the example drug response data, cell line gene expression image data, and drug descriptor image data
    logger.info('Loading response data.')
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
    logger.info('Loading cancer features')
    canc_feat = cancer_feat(feat)
    keys = list(canc_feat.keys())
    feat_c = canc_feat[keys[0]]
    if len(keys)>1:
        for i in range(1, len(keys)):
            feat_c = feat_c.join(canc_feat[keys[i]], lsuffix=str('_'+keys[i-1]), rsuffix=str('_'+keys[i]), how='outer')
    #use only cell-lines with response data available
    feat_c = feat_c.loc[res['improve_sample_id'].values]
    #Drug features
    logger.info('Loading drug features')
    feat_d= drug_feat()
    feat_d = feat_d.loc[res['improve_chem_id'].values]

    #Concatenate cell-line and drug features
    feat_concat = pd.concat([feat_c.reset_index(drop=True), feat_d.reset_index(drop=True)], axis=1)
    
    # Normalize features excluding metadata columns
    scaler = StandardScaler()
    feat_concat = pd.DataFrame(scaler.fit_transform(feat_concat), columns=feat_concat.columns, index=feat_concat.index)
    
    # Create metadata DataFrame
    meta_data = res[['improve_sample_id', 'improve_chem_id', 'auc']]

    #Add AUC value
    #feat_concat.insert(loc = 0, column = 'AUC', value = meta_data['auc'])
    #feat_concat.insert(loc = 0, column = 'Cell_line', value = meta_data['improve_sample_id'])
    #feat_concat.insert(loc = 0, column = 'Drug', value = meta_data['improve_chem_id'])
    return feat_concat, meta_data, num_sample
    
def data_double_network(response_dir):
    # Load the example drug response data, cell line gene expression image data, and drug descriptor image data
    logger.info('Loading response data.')
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
    logger.info('Loading cancer features')
    canc_feat = cancer_feat(feat)
    keys = list(canc_feat.keys())
    feat_c = canc_feat[keys[0]]
    if len(keys)>1:
        for i in range(1, len(keys)):
            feat_c = feat_c.join(canc_feat[keys[i]], lsuffix=str('_'+keys[i-1]), rsuffix=str('_'+keys[i]), how='outer')
    #use only cell-lines with response data available
    feat_c = feat_c.loc[res['improve_sample_id'].values]
    # Normalize cancer features
    scaler = StandardScaler()
    feat_c = pd.DataFrame(scaler.fit_transform(feat_c), columns=feat_c.columns, index=feat_c.index)

    #Drug features
    logger.info('Loading drug features')
    feat_d= drug_feat()
    feat_d = feat_d.loc[res['improve_chem_id'].values]
    scaler = StandardScaler()
    feat_d = pd.DataFrame(scaler.fit_transform(feat_d), columns=feat_d.columns, index=feat_d.index)
 
    # Create metadata DataFrame
    meta_data = res[['improve_sample_id', 'improve_chem_id', 'auc']]

    return feat_c, feat_d, meta_data, num_sample


def multi_channel_prediction(feat, savedir, response_dir, model_params): 
    # Load the example drug response data, cell line gene expression image data, and drug descriptor image data
    if single_network:
        feat_concat, meta_data, num_sample=data_single_network(response_dir)
    else:
        feat_c, feat_d, meta_data, num_sample=data_double_network(response_dir)

    seeds = list(range(0,num_data_partitions))
    for seed in seeds:
        logger.info(f'** PARTITION {seed} **')
        # Create the directory for saving results
        result_dir = str('../Results/Prediction_On_Images/Regression_Prediction/'+ savedir +'/partition_'+str(seed))
        if os.path.exists(result_dir) and Path(str(result_dir+'/Prediction_Performance.txt')).is_file():
            logger.info("Skipping partition......")
            continue
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)
        np.random.seed(seed)
        rand_sample_ID = np.random.permutation(num_sample)
        fold_size = int(num_sample / num_fold)
        sampleID = {}
        sampleID['trainID'] = rand_sample_ID[range(fold_size * (num_fold - 2))]
        sampleID['valID'] = rand_sample_ID[range(fold_size * (num_fold - 2), fold_size * (num_fold - 1))]
        sampleID['testID'] = rand_sample_ID[range(fold_size * (num_fold - 1), num_sample)]
        if single_network:
            X_train = feat_concat.iloc[sampleID['trainID']].reset_index(drop=True)
            X_val = feat_concat.iloc[sampleID['valID']].reset_index(drop=True)
            X_test = feat_concat.iloc[sampleID['testID']].reset_index(drop=True)
            Y_train = meta_data['auc'].iloc[sampleID['trainID']].reset_index(drop=True)
            Y_val = meta_data['auc'].iloc[sampleID['valID']].reset_index(drop=True)
            Y_test = meta_data['auc'].iloc[sampleID['testID']].reset_index(drop=True)
            model, val_perf, train_perf = dnn_model(X_train, X_val, Y_train, Y_val, model_params) 
            test_perf = predict_hold_out(model, X_test, Y_test, model_params)
        else:
            X_train_cancer = feat_c.iloc[sampleID['trainID']].reset_index(drop=True)
            X_val_cancer = feat_c.iloc[sampleID['valID']].reset_index(drop=True)
            X_test_cancer = feat_c.iloc[sampleID['testID']].reset_index(drop=True)
            X_train_drug = feat_d.iloc[sampleID['trainID']].reset_index(drop=True)
            X_val_drug = feat_d.iloc[sampleID['valID']].reset_index(drop=True)
            X_test_drug = feat_d.iloc[sampleID['testID']].reset_index(drop=True)
            Y_train = meta_data['auc'].iloc[sampleID['trainID']].reset_index(drop=True)
            Y_val = meta_data['auc'].iloc[sampleID['valID']].reset_index(drop=True)
            Y_test = meta_data['auc'].iloc[sampleID['testID']].reset_index(drop=True)
            model, val_perf, train_perf = dnn_model_double_network(X_train_cancer, X_val_cancer, X_train_drug, X_val_drug, Y_train, Y_val, model_params)
            test_perf = predict_hold_out_double_network(model, X_test_cancer, X_test_drug, Y_test, model_params)

        perf = np.empty((3, 4))
        perf.fill(np.nan)
        perf = pd.DataFrame(perf, columns=['R2', 'MSE', 'pCor', 'sCor'],
                    index=['train', 'val', 'test'])
        perf.loc['train', 'R2'] = train_perf['r2']
        perf.loc['train','MSE'] = train_perf['rmse']
        perf.loc['train','pCor'] = train_perf['pCor']
        perf.loc['train','sCor'] = train_perf['sCor']

        perf.loc['val','R2'] = val_perf['r2']
        perf.loc['val','MSE'] = val_perf['rmse']
        perf.loc['val','pCor'] = val_perf['pCor']
        perf.loc['val','sCor'] = val_perf['sCor']

        perf.loc['test','R2'] = test_perf['r2']
        perf.loc['test','MSE'] = test_perf['rmse']
        perf.loc['test','pCor'] = test_perf['pCor']
        perf.loc['test','sCor'] = test_perf['sCor']

        perf.to_csv(result_dir + '/Prediction_Performance.txt', header=True, index=True, sep='\t', line_terminator='\r\n')

        output = {}
        output['best_epochs']=val_perf['best_epoch']
        output['model_params'] = model_params
        output['train_perf'] = train_perf
        output['val_perf'] = val_perf
        output['test_perf'] = test_perf
        output['feat'] = feat
        output['size_of_response'] = num_sample
        output['study']= study
        output['single_network'] = single_network
        savename='output.json'
        with open(result_dir + "/"+ savename, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)




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

#feats = [['mut_inter1_mean'], ['mut_inter2_mean'], ['mut_inter3_mean'], ['mut_inter4_mean'], ['mut_inter5_mean'],
#         ['mut_del_inter1_mean'], ['mut_del_inter2_mean'], ['mut_del_inter3_mean'], ['mut_del_inter4_mean'], ['mut_del_inter5_mean']]
num_data_partitions = 20  # Change this if needed
num_fold = 10
response_dir = '../csa_data/raw_data/y_data/response.tsv'
drug_specific = False
study = 'CTRPv2' #CTRPv2 or GDSCv2 or CCLE
single_network=False
model_params={'batch_size':32,
              'test_batch_size':32,
              'seed':0,
              'lr':0.0001,
              'n_epochs':100,
              #'h_sizes':[512, 256, 125, 64, 32], #-single DNN
              'h_sizes':[512,256, 125], # double network
              'dropout':0.2,
              'patience': 10,
              'max_half_num_batch':150,
              'num_data_partitions': num_data_partitions,
              'single_network': single_network,
              'drug_specific': drug_specific,
              'sigmoid_activation':False
}

#feats=[['ge']]
#feats = [['mut_bin_del'], ['mut_bin'],['mut_inter1_mean'], ['mut_inter2_mean'], ['mut_inter3_mean'], ['mut_inter4_mean'], ['mut_inter5_mean']]
#feats=[['mut_del_inter1_mean'], ['mut_del_inter2_mean'], ['mut_del_inter3_mean'], ['mut_del_inter4_mean'], ['mut_del_inter5_mean']]
feats=[['mut_del_hubScore_inter3'], ['mut_del_hubScore_inter2'], ['mut_del_hubScore_inter1'],['mut_del_hubScore_inter4'], ['mut_del_hubScore_inter5'],
['mut_hubScore_inter3'], ['mut_hubScore_inter2'], ['mut_hubScore_inter1'], ['mut_hubScore_inter4'], ['mut_hubScore_inter5']]

if drug_specific:
    logger.info('Start training workflow for drug specific')
    drugs = drug_selection(R2_filter=True, num_drugs=10)
    for feat in feats:
        for drug in drugs:
            savename = '_'.join(feat)
            savedir = str('DNN_mutations/drug_specific_mutations/'+study+'/'+savename)  ## CHANGE THIS
            multi_channel_prediction_drug_specific(drug, feat, savedir, response_dir)

else:
    logger.info('Start training workflow for pan drug pan cancer')
    for feat in feats:
        logger.info(f'Start training for {feat}')
        savename = '_'.join(feat)
        #savename='ge_sn_hp2'
        savedir = str('DNN_mutations/pan_drug_mutations/Double_network/'+study+'/'+savename)  ## CHANGE THIS
        multi_channel_prediction(feat, savedir, response_dir, model_params)



