# **PhaseOmniMapper**

The implementation of the paper "PhaseOmniMapper : computational deconvolution of LLPS mechanism from proteome-wide Scaffold/Client classification to key residue detection via hybrid architectures".

# **Requirements**
- python           3.8.18 
* tensorflow-gpu   2.7.0
+ torch            2.4.1
- Biopython
* Pandas
+ numpy
- matplotlib
* sklearn
+ openssl          3.0.16

# **Usage**

We implemented the PhaseOmniMapper model using the Tensorflow 2.7.0 and Torch 2.4.1 deep learning frameworks and trained the model on an Nvidia RTX4090 GPU.

# **Training the model**
1. To train a PhaseOmniMapper model to predict scaffolds, you need to prepare the PDB file in CSV format using PDBProcess.py. You can also obtain the feature matrix obtained using ProtT5, which can be obtained from https://github.com/agemagician/ProtTrans/blob/master/Embedding/TensorFlow/Advanced/ProtT5-XL-UniRef50.ipynb.

2. Train-predict.py is used to train and predict scaffold proteins. train-client.py is used to train and predict client proteins and LLPS proteins containing IDR regions. train-idr-residue.py is used to train and predict IDR residues.

3. Client and LLPS proteins containing IDR regions, RF+RFE.py was used to select the optimal feature space before training.

4. When constructing IDR-related datasets, IDR1.py and IDR2.py are used to process the scaffold protein and the client protein to obtain a dataset containing IDR regions and mark the IDR residue positions.

# **Using trained model for human α-synuclein prediction**

1. The PDB file of human α-synuclein was obtained using esmfold.py, and the feature matrix was extracted using ProtT5, which was then integrated with the obtained CSV file of physicochemical properties as input.
2. Predicting LLPS propensity of human α-synuclein using predict.py.
3. The trained mean_std.npz and onehot_encoder.pkl are in the IDR folder.

# **Notification of commercial use**
Commercialization of this product is prohibited without our permission.
