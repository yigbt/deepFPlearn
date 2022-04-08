import numpy as np

# Datatype that is used for fingerprint vectors stored inside
# dataframe used during the program
df_fp_numpy_type = np.bool8

# Do we need copies when creating numpy matrices from the pandas dataframes
# for training? Everything seems to work fine with False and it saves memory.
numpy_copy_values = False

# Autoencoder data type settings

# Type that is given as input numpy array to the network
# We need to carefully test the performance of the training here
# The big dsstox dataset runs out of mem on Patrick's machine when
# we use floats here.
# Also, we possibly should give the FNN a float type fp
ac_fp_numpy_type = np.bool8

# Type used for the compressed fp which is an inner layer of the ac
# an MUST be of float type. On Patrick's machine, (GPU) tensorflow layers
# are float32 per default. Not sure if float64 can be used with GPU
# training and if it enhances results. Needs to be evaluated.
ac_fp_compressed_numpy_type = np.bool8

# Feedforward Neuronal Network data type settings
#
# The sections below define the types of the numpy matrices used for training
# the FNNs. In short:
# XX_fp_numpy_type: used for uncompressed fingerprint tensors
# XX_fp_compressed_numpy_type: used for compressed fingerprint tensors
# XX_target_numpy_type: used for the target tensors

nn_fp_numpy_type = np.float32
nn_fp_compressed_numpy_type = np.float32
nn_target_numpy_type = np.float32

nn_multi_fp_numpy_type = np.float32
nn_multi_fp_compressed_numpy_type = np.float32
nn_multi_target_numpy_type = np.float32

# Training settings

# Training settings of the AC that were magic numbers in the code before.
ac_train_min_delta = 0.001
ac_train_check_period = 5
ac_train_patience = 10

# Training settings of the FNN that were magic numbers in the code before.
nn_train_min_delta = 0.0001
nn_train_check_period = 5
nn_train_patience = 10
