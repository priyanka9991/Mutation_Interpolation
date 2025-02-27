# Mutation_Interpolation
Mathematical interpolation of sparse mutation data for anti-cancer drug response prediction.  
The sparse representation of mutation across datasets hinders predictive model performance. In this work, we implement a sparse mutation interpolation framework to enhance drug response prediction. By leveraging mathematical interpolation techniques, our approach imputes sparse mutation information to generate a more complete representation of the genomic landscape. We conduct the analysis on three cell line drug screening datasets: the Cancer Therapeutics Response Portal v2 (CTRP) GDSCv2 and CCLE. In drug response modeling, cell lines are represented by binary mutation data, where a value of '1' indicates the presence of a mutation in a gene.  Analysis focuses on 1,936 genes selected as the union of genes from the Library of Integrated Network-Based Cellular Signatures (LINCS), oncogenes, and COSMIC genes. Drug data is represented by Mordred molecular fingerprints, and 1,600 features with maximum variation were selected for modeling. We leverage protein-protein interaction (PPI) networks to propagate mutation values. To address this, we leverage protein-protein interaction (PPI) networks to propagate mutation values. This approach assigns values to non-mutated genes based on their proximity to mutated genes within the network, effectively mitigating the sparsity issue and enhancing the utility of the data for predictive modeling. Four machine learning models are used to demonstrate the effectiveness of the approach:  
1. LightGBM
2. Single network deep neural network (DNN) model with concatenated cancer and drug features
3. Double netwrok DNN with separate subnetworks for cancer and drug features
4. IGTD - image representations of cancer and drug features

# Scripts

1. mutation_del_interpolation.py - Generate interpolated mutation data. The parameter rounds (in line 89) determines the maximum number of iterations to run the algorithm. For example, when rounds=4, four different interpolated mutation data will be saved for rounds = 1, 2, 3 and 4.
2. lgbm prediction - Train and evaluate LightGBM model on the interpolated data.
3. dnn_prediction.py - Trains and evaluate neural network models on the interpolated data. When the parameter single_network=True, DNN model with single subnetwork is trained; when single_network=False, DNN model with double network is trained.  
4. multiomics_image.py - Generate IGTD images from tabular mutation data.
5. Multi_channel_prediction.py  - Trains and evaluates IGTD model on images generated from mutation data.



