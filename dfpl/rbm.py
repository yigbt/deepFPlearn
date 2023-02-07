import os.path
from os.path import basename
import pandas as pd
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import math

from dfpl import callbacks
from dfpl import options
from dfpl import history as ht
from dfpl import settings


class RBMLayer(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, name="RBMLayer", **kwargs):
        super(RBMLayer, self).__init__(name=name, **kwargs)
        self._input_size = input_size
        self._output_size = output_size
        self._name = name
        # self._activation_function = activation_function

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self._output_size),
            initializer="random_normal",
            trainable=True, name=f"w_{self._name}"
        )
        self.hb = self.add_weight(
            shape=(self._output_size,), initializer="random_normal", trainable=True, name=f"hb_{self._name}"
        )
        self.vb = self.add_weight(
            shape=(input_shape[-1],), initializer="random_normal", trainable=True, name=f"vb_{self._name}"
        )

    def __prob_h_given_v(self, visible, w, hb):
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    def __prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    def __sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    def call(self, X):
        out = tf.nn.sigmoid(tf.matmul(X, self.w) + self.hb)
        hiddenGen = self.__sample_prob(self.__prob_h_given_v(X, self.w, self.hb))
        visibleGen = self.__sample_prob(self.__prob_v_given_h(hiddenGen, self.w, self.vb))
        h1 = self.__prob_h_given_v(visibleGen, self.w, self.hb)
        return out, visibleGen, hiddenGen, h1

    def get_config(self):
        return {"input_size": self._input_size,
                "output_size": self._output_size}


class RBM(tf.keras.Model):

    def __init__(self, input_size, output_size, name="RBM", **kwargs):
        super(RBM, self).__init__(name=name, **kwargs)
        # Define hyperparameters
        self._input_size = input_size
        self._output_size = output_size
        self.n_layer = round(math.log2(input_size / output_size))

        self.last_out = None

        # layers defination
        # encoding layers
        self.rbm_encoder_layers = []
        self.rbm_decoder_layers = []
        self.rbm_encoder1 = RBMLayer(self._input_size, int(self._input_size // 2), name="encoder_0")

        if self.n_layer > 1:
            for i in range(1, self.n_layer):
                factor_units = 2 ** (i + 1)
                self.rbm_encoder_layers.append(
                    RBMLayer(int(self._input_size // factor_units) * 2, int(self._input_size // factor_units),
                             name=f"encoder_{i}"))

            # decoding layers
            factor_units = 2 ** (self.n_layer - 1)
            self.rbm_decoder1 = RBMLayer(int(input_size // factor_units) // 2, int(input_size // factor_units),
                                         name="decoder_0")

            for layer_num, i in enumerate(range(self.n_layer - 2, 0, -1)):
                factor_units = 2 ** i
                self.rbm_decoder_layers.append(
                    RBMLayer(int(input_size // (factor_units * 2)), int(input_size // factor_units),
                             name=f"decoder_{layer_num + 1}"))

            self.rbm_decoder_layers.append(
                RBMLayer(int(input_size // 2), int(input_size), name=f"decoder_{self.n_layer - 1}"))

    def call(self, X, training=None):
        outs = {}
        out, visibleGen, hiddenGen, h1 = self.rbm_encoder1(X)
        # out, visibleGen, hiddenGen, h1 = self.rbm_encoder(X)
        outs[f'encoder_{0}'] = (out, visibleGen, hiddenGen, h1)
        for i, layer in enumerate(self.rbm_encoder_layers):
            out, visibleGen, hiddenGen, h1 = layer(hiddenGen)
            outs[f'encoder_{i + 1}'] = (out, visibleGen, hiddenGen, h1)

        out, visibleGen, hiddenGen, h1 = self.rbm_decoder1(hiddenGen)
        outs[f'decoder_{0}'] = (out, visibleGen, hiddenGen, h1)
        for j, layer in enumerate(self.rbm_decoder_layers):
            out, visibleGen, hiddenGen, h1 = layer(hiddenGen)
            outs[f'decoder_{j + 1}'] = (out, visibleGen, hiddenGen, h1)
        return out, visibleGen, hiddenGen, h1, outs

    def train_step(self, data):
        X, y = data

        learning_rate = self.optimizer._decayed_lr('float32')
        out, visibleGen, hiddenGen, h1, out_dict = self(X, training=True)
        weights_dict = {}
        for l in self.layers:
            weights_dict[l.name] = l

        last_out = X
        for idx, (key, value) in enumerate(out_dict.items()):
            out, visibleGen, hiddenGen, h1 = value
            positive_grad = tf.matmul(tf.transpose(last_out), hiddenGen)
            negative_grad = tf.matmul(tf.transpose(visibleGen), h1)
            # loss = tf.reduce_mean(tf.square(y, visibleGen))
            # loss = self.compiled_loss(y, visibleGen)

            # update the weights of the model based on the loss and LR.
            curr_layer = weights_dict[key]
            curr_layer.weights[0].assign(curr_layer.weights[0] + learning_rate * \
                                         (positive_grad - negative_grad) / tf.cast(tf.shape(last_out)[0],
                                                                                   dtype=tf.float32))
            curr_layer.weights[1].assign(curr_layer.weights[1] + learning_rate * tf.math.reduce_mean(hiddenGen - h1, 0))
            curr_layer.weights[2].assign(
                curr_layer.weights[2] + learning_rate * tf.math.reduce_mean(last_out - visibleGen, 0))

            last_out = hiddenGen
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        out, visibleGen, hiddenGen, h1, out_dict = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, hiddenGen, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, hiddenGen)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def define_rbm_model(opts: options.Options,
                     input_size: int = 2048,
                     encoding_dim: int = 128,
                     my_loss: str = "binary_crossentropy",
                     my_lr: float = 0.0288,
                     my_decay: float = 0.0504) -> (Model):
    """
    This function provides an autoencoder model to reduce a certain input to a compressed version.

    :param encoding_dim: Size of the compressed representation. Default: 85
    :param input_size: Size of the input. Default: 2048
    :param my_loss: Loss function, see tensorflow.keras Loss functions for potential values. Default: binary_crossentropy
    :param my_lr:
    :param my_decay:
    :return: a tuple of autoencoder and encoder models
    """
    if opts.aeOptimizer == "Adam":
        ac_optimizer = tf.keras.optimizers.Adam(learning_rate=opts.aeLearningRate, decay=opts.aeLearningRateDecay)
    elif opts.aeOptimizer == "SGD":
        ac_optimizer = tf.keras.optimizers.SGD(learning_rate=opts.aeLearningRate, decay=opts.aeLearningRateDecay)

    rbm_model = RBM(input_size=input_size,
                    output_size=encoding_dim)
    rbm_model.compile(optimizer=ac_optimizer, loss=tf.keras.losses.MeanSquaredError())

    return rbm_model


def train_full_rbm(df: pd.DataFrame, opts: options.Options) -> Model:
    """
    Train an autoencoder on the given feature matrix X. Response matrix is only used to
    split meaningfully in test and train data set.

    :param opts: Command line arguments as defined in options.py
    :param df: Pandas dataframe that contains the smiles/inchi data for training the autoencoder
    :return: The encoder model of the trained autoencoder
    """

    # Set up the model of the AC w.r.t. the input size and the dimension of the bottle neck (z!)
    rbm = define_rbm_model(opts, input_size=opts.fpSize,
                               encoding_dim=opts.encFPSize)
    # callback_list = callbacks.autoencoder_callback(checkpoint_path=rbm_weights_file, opts=opts)

    # define output file for autoencoder and encoder weights
    if opts.ecWeightsFile == "":
        logging.info("No RBM encoder weights file specified")
        base_file_name = os.path.splitext(basename(opts.inputFile))[0]
        logging.info(f"RBM weights will be saved in {base_file_name}.rbm{opts.encFPSize}.hdf5")
        rbm_weights_file = os.path.join(opts.outputDir, base_file_name + f".rbm{opts.encFPSize}.hdf5")
        ec_weights_file = os.path.join(opts.outputDir, base_file_name + ".rbm_encoder.weights.hdf5")
    else:
        logging.info(f"RBM encoder will be saved in {opts.ecWeightsFile}")
        base_file_name = os.path.splitext(basename(opts.ecWeightsFile))[0]
        rbm_weights_file = os.path.join(opts.outputDir, opts.ecWeightsFile)

    # collect the callbacks for training
    callback_list = callbacks.autoencoder_callback(checkpoint_path=rbm_weights_file, opts=opts)

    # Select all fps that are valid and turn them into a numpy array
    # This step is crucial for speed!!!
    fp_matrix = np.array(df[df["fp"].notnull()]["fp"].to_list(),
                         dtype=settings.ac_fp_numpy_type,
                         copy=settings.numpy_copy_values)
    logging.info(f"Training RBM on a matrix of shape {fp_matrix.shape} with type {fp_matrix.dtype}")

    # split data into test and training data
    x_train, x_test = train_test_split(fp_matrix,
                                       test_size=0.2,
                                       random_state=42)
    x_train = tf.cast(tf.where(x_train, 1, 0), tf.float32)
    x_test = tf.cast(tf.where(x_test, 1, 0), tf.float32)
    logging.info(f"RBM train data shape {x_train.shape} with type {x_train.dtype}")
    logging.info(f"RBM test data shape {x_test.shape} with type {x_test.dtype}")

    auto_hist = rbm.fit(x=x_train, y=x_train,
                            callbacks=callback_list,
                            epochs=opts.aeEpochs,
                            batch_size=opts.aeBatchSize,
                            verbose=opts.verbose,
                            validation_data=(x_test, x_test))
    rbm.summary(print_fn=logging.info)
    ht.store_and_plot_history(base_file_name=os.path.join(opts.outputDir, base_file_name + ".RBM"),
                              hist=auto_hist)
    rbm.save_weights(rbm_weights_file)

    # tf.saved_model.save(encoder,'/home/soulios/deepFPlearn-master/example/results_train/')
    logging.info(f"RBM weights stored in file: {rbm_weights_file}")

    return rbm


def compress_fingerprints(dataframe: pd.DataFrame,
                          encoder: Model, layer_num: int) -> pd.DataFrame:
    """
    Adds a column of the compressed version of the fingerprints to the original dataframe.

    :param dataframe: Dataframe containing a column named 'fp' with the fingerprints
    :param encoder: The trained autoencoder that is used for compressing the fingerprints
    :return: The input dataframe extended by a column containing the compressed version of the fingerprints
    """
    logging.info("Adding compressed fingerprints")
    idx = dataframe[dataframe["fp"].notnull()].index
    fp_matrix = np.array(dataframe[dataframe["fp"].notnull()]["fp"].to_list(),
                         dtype=settings.ac_fp_numpy_type,
                         copy=settings.numpy_copy_values)
    logging.info(f"Using input matrix of shape {fp_matrix.shape} with type {fp_matrix.dtype}")
    logging.info("Compressed fingerprints are added to input dataframe.")
    fp_matrix = tf.cast(tf.where(fp_matrix, 1, 0), tf.float32)
    out, visibleGen, hiddenGen, h1, out_dict = encoder.predict(fp_matrix)
    out, visibleGen, hiddenGen, h1 = out_dict[f'encoder_{layer_num}']
    dataframe['fpcompressed'] = pd.DataFrame({'fpcompressed': [s for s in hiddenGen]}, idx)

    return dataframe
