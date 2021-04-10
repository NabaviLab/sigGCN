# sigGCN

## 1. Generate the gene adjacency network

If the gene-gene network is not available, use the getADJ.py to generate the sparse gene adjacency network. 

Five varables in the function:

@dataPath: directory of input expr matrix

@networkPath：directory of STRING or BioGrid network

@dataset：which dataset to load, get a list of gene official names

@net：which network to use, STRING or BioGrid

@pathToSave：directory to save the adj matrix

Run the file getADJ.py: 

python getADJ.py --dataPath="data" --networkPath="data" --dataset="Zhengsorted" net="String"  --pathToSave="data/Zhengsorted"



## 2. Load data and train the model

Six variables should be provided.

@dirData, type=str, "directory of cell x gene matrix"

@dataset, type=str, "dataset to load"

@dirAdj, type = str, "directory of adj matrix"

@dirLabel, type = str, "directory of adj matrix"

@outputDir, type = str, "directory to save results"

@saveResults, type=int, default=0, "whether or not save the results, use 1 to save and 0 to not save"

To train the model, run the file:

python siggcn.py --dirData="data/" --dataset="Zhengsorted" --dirAdj="data/Zhengsorted/" --dirLabel="data/Zhengsorted/" --outputDir="data/output" --saveResults=0



### 2.1 load data

1. The data matrix (cell x gene expression matrix), adj matrix, labels, and a fixed shuffle index are loaded first. 
2. Then use the "down_gene" choose the top N genes by variance. 
3. Split the data into training, validation, and testing dataset.
4. Generate the data loader for training and evaluation.

### 2.2 Train the model

The dataset will be split into 80% as training, 10% as validaton, 10% as testing. 

### 2.3 Save results

 if use @saveResults=1, the results will be saved into the output folder.





















