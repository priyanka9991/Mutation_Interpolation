import pandas as pd
import os
from IGTD_Functions import min_max_transform, select_features_by_variation, multi_table_to_image, table_to_image
import numpy as np
import random


def create_ppi_dict(ppi_int, all_genes):
    ppi_dict = {}
    for _, row in ppi_int[ppi_int['#node1'].isin(all_genes) & ppi_int['node2'].isin(all_genes)].iterrows():
        ppi_dict.setdefault(row['#node1'], {})[row['node2']] = row['combined_score']
    return ppi_dict

def interpolate_cell_line_score(cell_line, ppi_dict, hub_score=False):
    mut_genes = cell_line.index[cell_line != 0].tolist()
    all_genes = cell_line.index.tolist()
    
    interpolated_values = {}
    for gene in all_genes:
        if gene not in mut_genes:
            if score_type=='max':
                max_score = max((ppi_dict.get(gene, {}).get(mut_gene, 0) * float(cell_line[mut_gene]) for mut_gene in mut_genes), default=0)
            if score_type=='mean':
                scores = [(ppi_dict.get(gene, {}).get(mut_gene, 0) * float(cell_line[mut_gene])) for mut_gene in mut_genes]
                scores=[e for i, e in enumerate(scores) if e != 0]
                max_score = 0 if len(scores)==0 else np.mean(scores)     
            if max_score!=0 and hub_score==True:
                len_non_zero=len(set(ppi_dict.get(gene, {}).keys()).intersection(set(mut_genes)))
                max_score=max_score*(len_non_zero/len(mut_genes))
            interpolated_values[gene] = max_score
    
    return interpolated_values

selected_genes=pd.read_table('../csa_data/raw_data/x_data/selected_genes.txt', header=None)
gene_subset = list(selected_genes[0].values)
gene_subset.insert(0, 'Cell-line')
selected_cl=pd.read_table('../csa_data/raw_data/x_data/selected_cl.txt', header=None)
common_cl = list(selected_cl[0].values)

#file = 'Mutation_AID_long_format_isDel.csv'
#mut_long= pd.read_table(str('../csa_data/raw_data/x_data/'+ file))

filter_del = True #### CHANGE IF NEEDED *****
file='Mutation_AID_binary_isDel.csv'
mut_b = pd.read_csv(str('../csa_data/raw_data/x_data/'+file))
gene = mut_b['Gene_symbol']
isdel = mut_b['isDeleterious']
mut_b = mut_b.drop(mut_b.columns[list(range(0,16))], axis=1)
mut_b.index=gene
mut_b.index.name=None
mut_b.columns= mut_b.iloc[0].values
mut_b = mut_b.iloc[1:,:]
if filter_del:
    ind_del = np.where(isdel==True)[0]-1
    mut_b=mut_b.iloc[ind_del]
mut_b = mut_b.T
mut_b = mut_b.loc[:,~mut_b.columns.duplicated()] # Remove duplicated columns
mut_b = mut_b[~mut_b.index.duplicated(keep='first')] # Remove rows with duplicated cell-lines
# We need to check for all zero columns
mut_b = mut_b.loc[:, ~(mut_b == 0).all()]
mut_b.insert(loc=0, column='Cell-line', value=mut_b.index.values)
mut_b=mut_b.reset_index(drop=True)
mut_b = mut_b[gene_subset].set_index(['Cell-line'])
mut_b.index.name=None
#data[f]=mut_b[gene_subset].set_index(['Cell-line'])

ppi_int = pd.read_table('../csa_data/raw_data/x_data/ppi_interactions.tsv')
gene_subset = list(selected_genes[0].values)

ppi_int=ppi_int[ppi_int['#node1'].isin(gene_subset)].reset_index(drop=True)
ppi_int=ppi_int[ppi_int['node2'].isin(gene_subset)].reset_index(drop=True)
#genes not in ppi interacions
ppi_genes=ppi_int['#node1'].unique()
#gene_not_available = list(set(gene_subset) - set(ppi_genes))
#mut_b=mut_b.drop(columns=gene_not_available)


# Create PPI dictionary once
all_genes = mut_b.columns.tolist()
ppi_dict = create_ppi_dict(ppi_int, all_genes)

mut_b_corrected = mut_b.copy()

if filter_del:
    del_status='del'
else:
    del_status='all'
score_type='mean' # max or mean
rounds=5
hub_score=True
for round in range(rounds):
    interpolated_data = {cl: interpolate_cell_line_score(mut_b_corrected.loc[cl], ppi_dict, hub_score) for cl in mut_b_corrected.index}
    mut_b_corrected.update(pd.DataFrame.from_dict(interpolated_data, orient='index'))
    mut_b_corrected.to_csv(str('../csa_data/raw_data/x_data/mut_ppi_combined_round_'+str(round+1)+'_'+score_type+'_'+del_status+'.csv'))

#ppi_genes.to_csv('../csa_data/raw_data/x_data/ppi_genes.txt')

#ppi_genesl = list(ppi_genes)
#with open('../csa_data/raw_data/x_data/ppi_genes.txt', 'w') as f:
#    for i in ppi_genesl:
#        f.write(i+'\n')

