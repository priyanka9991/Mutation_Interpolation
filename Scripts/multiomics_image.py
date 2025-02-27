import pandas as pd
import os
from IGTD_Functions import min_max_transform, select_features_by_variation, multi_table_to_image, table_to_image
import numpy as np
import random

def cancer_feat_by_var(feat = ['ge'], max_step=30000, val_step=300, image_dist_method = 'Euclidean', dim = [100,100], savedir='Cancer'):
    feat_data_dict = {'ge': 'cancer_gene_expression.tsv', 'cnv':'cancer_copy_number.tsv', 'mut_count':'cancer_mutation_count.tsv','mut_bin':'Mutation_AID_binary_isDel.csv',
                    'cnv_disc':'cancer_discretized_copy_number.tsv', 'meth':'cancer_DNA_methylation.tsv', 'mut_long':'Mutation_AID_long_format_isDel.csv'}
    # Select features my maximum variation
    file = feat_data_dict['ge']
    ge= pd.read_table(str('../csa_data/raw_data/x_data/'+ file))
    ge.columns = ge.iloc[1] #Use only gene symbol column
    ge = ge.drop([0,1])
    data = ge.rename(columns={ge.columns[0]:'Cell-line'}).reset_index(drop=True)
    cl = data['Cell-line']
    data = data.drop(columns=['Cell-line'])
    id = select_features_by_variation(data.apply(pd.to_numeric), variation_measure='var', num=dim[0]*dim[1])
    data = data.iloc[:, id]
    data.insert(loc=0, column='Cell-line', value=cl)
    data=data.set_index(['Cell-line'])
    data.index.name = None
    data= data.apply(pd.to_numeric)
    # Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively.
    norm_data = min_max_transform(data.values)
    norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)
    fea_dist_method = 'Euclidean'
    image_dist_method = image_dist_method
    error = 'abs'
    save_image_size = 3
    result_dir = str('../Results/Table_To_Image_Conversion/'+savedir)
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(norm_data, [dim[0], dim[1]], fea_dist_method, image_dist_method, save_image_size,
                max_step, val_step, result_dir, error)

def cancer_features(feat = ['ge','mut_bin_del', 'cnv'], max_step=30000, val_step=300, weight_list=[0.33,0.33,0.33], image_dist_method = 'Euclidean', fea_dist_method_list=('Euclidean', 'Euclidean'), dim = [100,100], GeneSubsetMethod = 'lincs_onco', savedir='Cancer', cl_subset=None):
    
    ind = [i for i, e in enumerate(weight_list) if e != 0]
    feat = [feat[index] for index in ind]
    weight_list = [weight_list[index] for index in ind]    
    fea_dist_method_list = tuple([fea_dist_method_list[index] for index in ind])
    print(fea_dist_method_list)
    feat_data_dict = {'ge': 'cancer_gene_expression.tsv', 'cnv':'cancer_copy_number.tsv', 'mut_count':'cancer_mutation_count.tsv','mut_bin_del':'Mutation_AID_binary_isDel.csv',
                    'mut_bin':'Mutation_AID_binary_isDel.csv','cnv_disc':'cancer_discretized_copy_number.tsv', 'meth':'cancer_DNA_methylation.tsv','mut_long':'Mutation_AID_long_format_isDel.csv',
                    'mut_del_inter3':'mut_del_corrected_ppi_round_3.csv','mut_del_inter2':'mut_del_corrected_ppi_round_2.csv','mut_del_inter1':'mut_del_corrected_ppi_round_1.csv',
                    'mut_inter3':'mut_corrected_ppi_round_3.csv','mut_inter2':'mut_corrected_ppi_round_2.csv','mut_inter1':'mut_corrected_ppi_round_1.csv',
                    'mut_del_hubScore_inter3':'mut_corrected_ppi_del_hubScore_round_3.csv', 'mut_del_hubScore_inter2':'mut_corrected_ppi_del_hubScore_round_2.csv', 'mut_del_hubScore_inter1':'mut_corrected_ppi_del_hubScore_round_1.csv',
                    'mut_hubScore_inter3':'mut_corrected_ppi_hubScore_round_3.csv', 'mut_hubScore_inter2':'mut_corrected_ppi_hubScore_round_2.csv', 'mut_hubScore_inter1':'mut_corrected_ppi_hubScore_round_1.csv',
                    'mut_hubScore_inter4':'mut_corrected_ppi_hubScore_round_4.csv', 'mut_hubScore_inter5':'mut_corrected_ppi_hubScore_round_5.csv',
                    'mut_del_hubScore_inter4':'mut_corrected_ppi_del_hubScore_round_4.csv', 'mut_del_hubScore_inter5':'mut_corrected_ppi_del_hubScore_round_5.csv',
                    'mut_inter1_mean':'mut_ppi_combined_round_1_mean_all.csv','mut_inter2_mean':'mut_ppi_combined_round_2_mean_all.csv','mut_inter3_mean':'mut_ppi_combined_round_3_mean_all.csv',
                    'mut_inter4_mean':'mut_ppi_combined_round_4_mean_all.csv','mut_inter5_mean':'mut_ppi_combined_round_5_mean_all.csv',
                    'mut_del_inter1_mean':'mut_ppi_combined_round_1_mean_del.csv','mut_del_inter2_mean':'mut_ppi_combined_round_2_mean_del.csv','mut_del_inter3_mean':'mut_ppi_combined_round_3_mean_del.csv',
                    'mut_del_inter4_mean':'mut_ppi_combined_round_4_mean_del.csv','mut_del_inter5_mean':'mut_ppi_combined_round_5_mean_del.csv'}


    # Gene subset selection - LINCS + ONCOGENES
    selected_genes=pd.read_table('../csa_data/raw_data/x_data/selected_genes.txt', header=None)
    gene_subset = list(selected_genes[0].values)
    gene_subset.insert(0, 'Cell-line')
    selected_cl=pd.read_table('../csa_data/raw_data/x_data/selected_cl.txt', header=None)
    common_cl = list(selected_cl[0].values)
    if cl_subset is not None:
        common_cl=list(set(cl_subset).intersection(set(common_cl)))

    #Number of zers to pad
    gene_tot = dim[0]*dim[1]
    num_to_add = int(gene_tot - len(gene_subset)+1)# number of zero to be added to make dim[0]xdim[1] images
    zero_names=[]
    [zero_names.append(str('zero_'+str(y))) for y in range(num_to_add)]
    zer0 = pd.DataFrame(0, index=common_cl, columns=zero_names) 

    #feature_extraction
    data={}
    for f in feat:
        if f=='mut_bin_del_ge':
            #GE
            file = feat_data_dict['ge']
            ge= pd.read_table(str('../csa_data/raw_data/x_data/'+ file))
            ge.columns = ge.iloc[1] #Use only gene symbol column
            ge = ge.drop([0,1])
            ge = ge.rename(columns={ge.columns[0]:'Cell-line'}).reset_index(drop=True)
            #Mut_del
            file = feat_data_dict['mut_bin_del']
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
            mut_b = mut_b.loc[:, ~(mut_b == 0).all()] ## Check this
            mut_b.insert(loc=0, column='Cell-line', value=mut_b.index.values)
            mut_b=mut_b.reset_index(drop=True)
            gene_com = list(set(mut_b.columns.values[1:]).intersection(ge.columns.values[1:]))
            cl_com = list(set(mut_b['Cell-line'].values).intersection(ge['Cell-line'].values))
            ge_n = ge[gene_com]
            ge_n.index=ge['Cell-line']
            ge_n.index.name=None
            ge_n=ge_n.loc[cl_com]
            mut_b_n = mut_b[gene_com]
            mut_b_n.index=mut_b['Cell-line']
            mut_b_n.index.name=None
            mut_b_n=mut_b_n.loc[cl_com]
            mut_b_n = mut_b_n.astype('int32')
            ge_n = ge_n.astype('float32')
            ge_mut = ge_n*mut_b_n
            # We need to check for all zero columns
            ge_mut = ge_mut.loc[:, ~(ge_mut == 0).all()] ## Check this
            ge_mut.insert(loc=0, column='Cell-line', value=ge_mut.index.values)
            ge_mut=ge_mut.reset_index(drop=True)
            data[f]=ge_mut[gene_subset].set_index(['Cell-line'])
            data[f].index.name = None
            data[f]=data[f].loc[common_cl]
            #data[f].loc[:, zero_names] = 0 #Pad zeros
            data[f] = data[f].apply(pd.to_numeric)
            continue

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
        if f=='meth':
            meth = pd.read_table(str('../csa_data/raw_data/x_data/'+file))
            meth.columns=meth.iloc[2]
            meth = meth.drop([0,1,2])
            meth = meth.rename(columns={meth.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=meth[gene_subset].set_index(['Cell-line'])
        if f=='cnv_disc':
            cnv = pd.read_table(str('../csa_data/raw_data/x_data/'+file))
            cnv.columns = cnv.iloc[0]
            cnv = cnv.drop([0,1])
            cnv = cnv.rename(columns={cnv.columns[0]:'Cell-line'}).reset_index(drop=True)
            data[f]=cnv[gene_subset].set_index(['Cell-line'])
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
        #data[f].loc[:, zero_names] = 0 #Pad zeros
        data[f] = data[f].apply(pd.to_numeric)

    data_pro=data.copy()
    ######## IGTD algorithm ##########
    num_row=num_col=dim[0] # Number of pixel rows and columns in image representation
    num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
    save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.

    max_step = max_step    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
    val_step = val_step      # The number of iterations for determining algorithm convergence. If the error reduction rate
                        # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

    # Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively.

    if len(data_pro.keys())==3:
        norm_data1 = min_max_transform(data_pro[list(data_pro.keys())[0]].values)
        norm_data1 = pd.DataFrame(norm_data1, columns=data_pro[list(data_pro.keys())[0]].columns, index=data_pro[list(data_pro.keys())[0]].index)
        norm_data1=pd.concat([norm_data1,zer0],axis=1)        
        norm_data2 = min_max_transform(data_pro[list(data_pro.keys())[1]].values)
        norm_data2 = pd.DataFrame(norm_data2, columns=data_pro[list(data_pro.keys())[1]].columns, index=data_pro[list(data_pro.keys())[1]].index)
        norm_data2=pd.concat([norm_data2,zer0],axis=1)        
        norm_data3 = min_max_transform(data_pro[list(data_pro.keys())[2]].values)
        norm_data3 = pd.DataFrame(norm_data3, columns=data_pro[list(data_pro.keys())[2]].columns, index=data_pro[list(data_pro.keys())[2]].index)
        norm_data3=pd.concat([norm_data3,zer0],axis=1)        
        image_dist_method = image_dist_method
        error = 'abs'
        result_dir = str('../Results/Table_To_Image_Conversion/'+savedir)
        os.makedirs(name=result_dir, exist_ok=True)
        # Run the IGTD algorithm using (1) three data tables together, (2) the Euclidean distance for calculating pairwise
        # feature distances and pariwise pixel distances and (3) the absolute function for evaluating the difference
        # between the feature distance ranking matrix and the pixel distance ranking matrix. Save the result in Test_1 folder.
        multi_table_to_image(norm_d_list=(norm_data1, norm_data2, norm_data3), weight_list=weight_list,
                            fea_dist_method_list=fea_dist_method_list, scale=[num_row, num_col],
                            image_dist_method=image_dist_method, save_image_size=save_image_size,
                            max_step=max_step, val_step=val_step, normDir=result_dir, error=error,
                            switch_t=0, min_gain=0.000001)

    if len(data_pro.keys())==2:
        norm_data1 = min_max_transform(data_pro[list(data_pro.keys())[0]].values)
        norm_data1 = pd.DataFrame(norm_data1, columns=data_pro[list(data_pro.keys())[0]].columns, index=data_pro[list(data_pro.keys())[0]].index)
        norm_data1=pd.concat([norm_data1,zer0],axis=1)        
        norm_data2 = min_max_transform(data_pro[list(data_pro.keys())[1]].values)
        norm_data2 = pd.DataFrame(norm_data2, columns=data_pro[list(data_pro.keys())[1]].columns, index=data_pro[list(data_pro.keys())[1]].index)
        norm_data2=pd.concat([norm_data2,zer0],axis=1)        

        image_dist_method = image_dist_method
        error = 'abs'
        result_dir = str('../Results/Table_To_Image_Conversion/'+savedir)
        os.makedirs(name=result_dir, exist_ok=True)
        # Run the IGTD algorithm using (1) three data tables together, (2) the Euclidean distance for calculating pairwise
        # feature distances and pariwise pixel distances and (3) the absolute function for evaluating the difference
        # between the feature distance ranking matrix and the pixel distance ranking matrix. Save the result in Test_1 folder.
        multi_table_to_image(norm_d_list=(norm_data1, norm_data2), weight_list=weight_list,
                            fea_dist_method_list=fea_dist_method_list, scale=[num_row, num_col],
                            image_dist_method=image_dist_method, save_image_size=save_image_size,
                            max_step=max_step, val_step=val_step, normDir=result_dir, error=error,
                            switch_t=0, min_gain=0.000001)

    if len(data_pro.keys())==1:
        norm_data1 = min_max_transform(data_pro[list(data_pro.keys())[0]].values)
        norm_data1 = pd.DataFrame(norm_data1, columns=data_pro[list(data_pro.keys())[0]].columns, index=data_pro[list(data_pro.keys())[0]].index)
        norm_data1=pd.concat([norm_data1,zer0],axis=1)        
        image_dist_method = image_dist_method
        error = 'abs'
        result_dir = str('../Results/Table_To_Image_Conversion/'+savedir)
        fea_dist_method = fea_dist_method_list[0]
        os.makedirs(name=result_dir, exist_ok=True)
        # Run the IGTD algorithm using (1) three data tables together, (2) the Euclidean distance for calculating pairwise
        # feature distances and pariwise pixel distances and (3) the absolute function for evaluating the difference
        # between the feature distance ranking matrix and the pixel distance ranking matrix. Save the result in Test_1 folder.
        table_to_image(norm_data1, [dim[0], dim[1]], fea_dist_method, image_dist_method, save_image_size,
                max_step, val_step, result_dir, error)

    # Run the IGTD algorithm using (1) two data tables, (2) the Pearson correlation coefficient for calculating
    # pairwise feature distances, (3) the Manhattan distance for calculating pariwise pixel distances, and
    # (4) the square function for evaluating the difference between the feature distance ranking matrix and
    # the pixel distance ranking matrix. Save the result in Test_2 folder.
    '''
    image_dist_method = 'Manhattan'
    error = 'squared'
    result_dir = '../Results/Table_To_Image_Conversion/Test_2'
    os.makedirs(name=result_dir, exist_ok=True)
    multi_table_to_image(norm_d_list=(norm_data1, norm_data2), weight_list=[0.5, 0.5],
                        fea_dist_method_list=('Pearson', 'Pearson'), scale=[num_row, num_col],
                        image_dist_method=image_dist_method, save_image_size=save_image_size,
                        max_step=max_step, val_step=val_step, normDir=result_dir, error=error,
                        switch_t=0, min_gain=0.000001)
    # table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
    #                max_step, val_step, result_dir, error)
    '''


############# Drugs ################
def drug_features(max_step=30000, val_step=300, image_dist_method = 'Euclidean', dim = [40,40], savedir='Drug'):
    # Mordred
    from IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation

    drug = pd.read_table('../csa_data/raw_data/x_data/drug_mordred.tsv')
    drug = drug.set_index('improve_chem_id')
    drug.index.name = None

    num_row = dim[0]    # Number of pixel rows in image representation
    num_col = dim[1]    # Number of pixel columns in image representation
    num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
    save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
    max_step = max_step    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
    val_step = val_step  # The number of iterations for determining algorithm convergence. If the error reduction rate
                    # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

    # Select features with large variations across samples
    id = select_features_by_variation(drug, variation_measure='var', num=num)
    data = drug.iloc[:, id]
    # Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively.
    norm_data = min_max_transform(data.values)
    norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

    # Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
    # distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
    # the pixel distance ranking matrix. Save the result in Test_1 folder.
    fea_dist_method = 'Euclidean'
    image_dist_method = image_dist_method
    error = 'abs'
    result_dir = str('../Results/Table_To_Image_Conversion/'+savedir)
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
                max_step, val_step, result_dir, error)

    # Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
    # (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
    # the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
    # Save the result in Test_2 folder.
    '''
    fea_dist_method = 'Pearson'
    image_dist_method = 'Manhattan'
    error = 'squared'
    norm_data = norm_data.iloc[:, :800]
    result_dir = '../Results/Table_To_Image_Conversion/Test_2'
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
                max_step, val_step, result_dir, error)
    '''

#'ge': 'cancer_gene_expression.tsv', 'cnv':'cancer_copy_number.tsv', 'mut_count':'cancer_mutation_count.tsv',
#                 'cnv_disc':'cancer_discretized_copy_number.tsv', 'meth':'cancer_DNA_methylation.tsv'
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

##Drug specific images - Ony use cell-line subsets for each drug
drug_specific=False
feat = ['ge']
if drug_specific:
    drugs = drug_selection(R2_filter=True, num_drugs=10)
    for drug in drugs:
        response_dir = '../csa_data/raw_data/y_data/response.tsv'
        res = pd.read_csv(response_dir, sep='\t', engine='c',
                      na_values=['na', '-', ''], header=0, index_col=None)
        res=res[res.source=='CTRPv2'].reset_index(drop=True) # Choose from 1 study
        res=res[res.improve_chem_id==drug].reset_index(drop=True) # Choose drug
        cl = res['improve_sample_id'].values.tolist()
        savedir=str('Drug_specific_data/'+drug+'/'+'_'.join(feat))
        cancer_features(feat = feat, max_step=50000, val_step=500, weight_list=[1], image_dist_method = 'Euclidean', fea_dist_method_list=['Euclidean'], dim = [45,45], 
                GeneSubsetMethod = 'Cosmic_cancer_actionable_onco_genes', savedir=savedir, cl_subset=cl)
        
#else:
    #cancer_features(feat = ['mut_inter1'], max_step=50000, val_step=500, weight_list=[1], image_dist_method = 'Euclidean', fea_dist_method_list=['Euclidean'], dim = [45,45], 
    #           GeneSubsetMethod = 'Cosmic_cancer_actionable_onco_genes', savedir='Mutation_interpolation_Oct_2024/mut_inter1')





### Function calls
# Features - ge, cnv, cnv_disc, mut_count, mut_bin_del, mut_bin_del_ge
#When using mut_bin_del make sure to use feat_dist_method as 'set'
#GeneSubsetMethod - Cosmic_cancer_actionable_onco_genes (dim - 50,50), lincs_onco (dim - 44, 44)

#cancer_features(feat = ['ge','mut_bin_del', 'cnv'], max_step=50000, val_step=500, weight_list=[0.33,0.33, 0.33], image_dist_method = 'Euclidean', fea_dist_method_list=('Euclidean', 'Euclidean', 'set'), dim = [50,50], GeneSubsetMethod = 'Cosmic_cancer_actionable_onco_genes', savedir='Cosmic_onco_50_50/Cancer_ge_mut_bin_del_cnv')
#cancer_features(feat = ['cnv','mut_bin_del'], max_step=50000, val_step=500, weight_list=[0.5,0.5], image_dist_method = 'Euclidean', fea_dist_method_list=('Euclidean', 'set'), dim = [50,50], GeneSubsetMethod = 'Cosmic_cancer_actionable_onco_genes', savedir='Cosmic_onco_50_50/Cancer_cnv_mut_bin_del')
#cancer_features(feat = ['mut_bin_del', 'cnv'], max_step=50000, val_step=500, weight_list=[0.5,0.5], image_dist_method = 'Euclidean', fea_dist_method_list=('set','Euclidean'), dim = [50,50], GeneSubsetMethod = 'Cosmic_cancer_actionable_onco_genes', savedir='Cosmic_onco_50_50/Cancer_mut_bin_del_cnv')
#cancer_features(feat = ['mut_inter1'], max_step=50000, val_step=500, weight_list=[1], image_dist_method = 'Euclidean', fea_dist_method_list=['Euclidean'], dim = [45,45], 
#                GeneSubsetMethod = 'Cosmic_cancer_actionable_onco_genes', savedir='Mutation_interpolation_Oct_2024/mut_inter1')
#ge, cnv, mut_count, mut_bin_del, mut_del_inter3, mut_del_inter2, mut_del_inter1, mut_inter3, mut_inter2, mut_inter1
#drug_features(max_step=30000, val_step=300, image_dist_method = 'Euclidean', dim = [40,40], savedir='Drug')
#cancer_feat_by_var(feat = ['ge'], max_step=50000, val_step=300, image_dist_method = 'Euclidean', dim = [50,50], savedir='Cancer_ge_max_var')

## ****************************************************
## ************* IMAGE GENERATION FOR 3D PLOTS *********
## ****************************************************


threeD_plot=False
if threeD_plot==True:
    feat = ['mut_inter1_mean', 'mut_inter2_mean', 'mut_inter3_mean', 'mut_inter4_mean', 'mut_inter5_mean',
            'mut_del_inter1_mean', 'mut_del_inter2_mean', 'mut_del_inter3_mean', 'mut_del_inter4_mean', 'mut_del_inter5_mean']
    #ge, cnv, mut_count, mut_bin_del, mut_del_inter3, mut_del_inter2, mut_del_inter1, mut_inter3, mut_inter2, mut_inter1
    # mut_del_hubScore_inter3, mut_del_hubScore_inter2, mut_del_hubScore_inter1, mut_hubScore_inter3, mut_hubScore_inter2, mut_hubScore_inter1
    x_all = [0, 100] #ratio for splits - contribution of the features
    #x_all=[100]
    fea_dist_method_list=[]
    [fea_dist_method_list.append('set') if f=='mut_bin_del' or f=='mut_bin' else fea_dist_method_list.append('Euclidean') for f in feat]
    fea_dist_method_list=tuple(fea_dist_method_list)


    split_ratios = []
    
    if len(feat)==3:
        for x in x_all:
            for y in x_all:
                z = 100 - x - y
                if 0 <= z <= 100:
                    split_ratios.append((x/100, y/100, z/100))
    if len(feat)==2:
        for x in x_all:
                z = 100 - x
                if 0 <= z <= 100:
                    split_ratios.append((x/100, z/100))
    if len(feat)==1:
        split_ratios=[(1.0)]

    for split in split_ratios: #[44:] :22, 22:44, 44:
        weight_list = list(split)
        ind = [i for i, e in enumerate(weight_list) if e != 0]
        feat_sel = [feat[index] for index in ind]
        weight_list = [weight_list[index] for index in ind]    
        fea_dist_method_list_sel = tuple([fea_dist_method_list[index] for index in ind])
        save_dir_name=''
        for i in range(len(feat_sel)):
            save_dir_name = str(save_dir_name +feat_sel[i]+'_'+str(weight_list[i]))
        savedir=str('Mutation_interpolation_Oct_2024/3d_plot_images/'+save_dir_name)
        result_dir = str('../Results/Table_To_Image_Conversion/'+savedir)

        if os.path.exists(result_dir):
            continue 
            
        if len(feat_sel)==1:
            cancer_features(feat = feat_sel, max_step=50000, val_step=500, weight_list=[1], image_dist_method = 'Euclidean', fea_dist_method_list=fea_dist_method_list_sel, 
                dim = [45,45], GeneSubsetMethod = 'Cosmic_cancer_actionable_onco_genes', savedir=savedir)
        else:
            cancer_features(feat = feat_sel, max_step=50000, val_step=500, weight_list=weight_list, image_dist_method = 'Euclidean', fea_dist_method_list=fea_dist_method_list_sel, 
                        dim = [45,45], GeneSubsetMethod = 'Cosmic_cancer_actionable_onco_genes', savedir=savedir)




feat = ['mut_inter1_mean', 'mut_inter2_mean', 'mut_inter3_mean', 'mut_inter4_mean', 'mut_inter5_mean',
         'mut_del_inter1_mean', 'mut_del_inter2_mean', 'mut_del_inter3_mean', 'mut_del_inter4_mean', 'mut_del_inter5_mean']
fea_dist_method_list=[]
[fea_dist_method_list.append('set') if f=='mut_bin_del' or f=='mut_bin' else fea_dist_method_list.append('Euclidean') for f in feat]
fea_dist_method_list=tuple(fea_dist_method_list)

for i in range(len(feat)):
    save_dir_name = str(feat[i]+'_1.0')
    savedir=str('Mutation_interpolation_Oct_2024/3d_plot_images/'+save_dir_name)
    result_dir = str('../Results/Table_To_Image_Conversion/'+savedir)
    fea_dist_method_list_sel = tuple([fea_dist_method_list[i]])
    cancer_features(feat = [feat[i]], max_step=50000, val_step=500, weight_list=[1], image_dist_method = 'Euclidean', fea_dist_method_list=fea_dist_method_list_sel, 
        dim = [45,45], GeneSubsetMethod = 'Cosmic_cancer_actionable_onco_genes', savedir=savedir)