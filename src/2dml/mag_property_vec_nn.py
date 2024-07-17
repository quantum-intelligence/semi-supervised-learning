#
# mag_property_vec_nn.property
# use NN to predit magnetic moment using property vector inputs
#
#######################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from import_functions import *
from six.moves.urllib.request import urlopen
import numpy as np
#import tensorflow as tf
from atomic_property import *

def gen_test_train(X_mag, t_mag):
    """ generate test and train data amd scale/normalize """
    N = len(t_mag)
    randindex = np.random.permutation(np.arange(N))
    trainsize = 0.75
    trN = np.int(np.floor(N*trainsize))
    testN = N-trN
    print(N, trN, testN)
    Xnn_train = X_mag.iloc[randindex[:trN],:].copy()
    Xnn_test = X_mag.iloc[randindex[trN:],:].copy()
    ynn_train = t_mag.iloc[randindex[:trN]].copy()
    ynn_test = t_mag.iloc[randindex[trN:]].copy()

    print(ynn_train.shape)
    print(ynn_test.shape)

    Xsnn_train, ysnn_train = scaledata_xy(Xnn_train, ynn_train)
    Xsnn_test, ysnn_test = scaledata_xy(Xnn_test, ynn_test)
    return Xsnn_train, ynn_train, Xsnn_test, ynn_test



def initiate_nn(Xsnn_train, ynn_train, Xsnn_test, ynn_test):
    """ setup NN model """

    TRAIN_X = Xsnn_train
    TRAIN_Y = ynn_train.values
    TEST_X = Xsnn_test
    TEST_Y = ynn_test.values

    print(TRAIN_X.shape)
    print(TRAIN_Y.shape)

    LEARNING_RATE = 0.01
    TRAINING_EPOCHS = 5000
    BATCH_SIZE = 10
    DISPLAY_STEP = 1

    N_HIDDEN_1 = 30
    N_HIDDEN_2 = 15
    #N_HIDDEN_3 = 16
    N_INPUT = TRAIN_X.shape[1]
    Nt_INPUT = TEST_X.shape[1]
    print('N_INPUT :', N_INPUT)
    Xsnn_train = pd.DataFrame(Xsnn_train)
    Xsnn_test = pd.DataFrame(Xsnn_test)

    return TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, LEARNING_RATE, TRAINING_EPOCHS, \
    BATCH_SIZE, DISPLAY_STEP, N_HIDDEN_1 ,N_HIDDEN_2, N_INPUT, Nt_INPUT, \
    Xsnn_train, Xsnn_test


def main( Xsnn_train,ynn_train, Xsnn_test,ynn_test ):
    # Load datasets.
    X_TRAIN = Xsnn_train  # Xsnn_train
    Y_TRAIN = ynn_train.values
    X_TEST = Xsnn_test
    Y_TEST = ynn_test

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[X_TEST.shape[1]])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    model_dir = '/Users/trevorrhone/Documents/Kaxiras/2DML/Alloys_ML/cnn_model_mag_pv'
    #del regressor
    regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                          hidden_units = [125, 45], #[25,15], #[10, 20, 10],
                                          activation_fn = tf.nn.tanh,
                                          dropout = 0.151,
                                          #optimizer = 'Adagrad',
                                          optimizer=tf.train.ProximalAdagradOptimizer(
                                              learning_rate=0.1, l1_regularization_strength=2.01),
                                          #learning_rate=0.1,
                                          #optimizer = tf.train.AdagradOptimizer(learning_rate=0.1),                                        l1_regularization_strength=0.001),
                                          model_dir=model_dir) #delete contents to resets calculation
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(X_TRAIN)},
      y=np.array(Y_TRAIN),
      num_epochs=None,
      shuffle=False)

    # Train model.
    regressor.train(input_fn=train_input_fn, steps=5000 )

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(X_TEST)},
      y=np.array(Y_TEST),
      num_epochs=1,
      shuffle=False)

    # Evaluate accuracy.
    accuracy_score = regressor.evaluate(input_fn=test_input_fn)['average_loss']
    print('REGRESSOR>EVALUATE', regressor.evaluate(input_fn=test_input_fn))
    print('accuracy_score', accuracy_score)

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify two new flower samples.

    new_samples = X_TEST.iloc[:,:].values
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

    predict_train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(X_TRAIN)},
      num_epochs=1,batch_size=50,
      shuffle=False)

    predictions = list(regressor.predict(input_fn=predict_input_fn))
    predictions_train = list(regressor.predict(input_fn=predict_train_input_fn))
    #predicted_classes = [p["classes"] for p in predictions]

    #print(
    #  'PREDICTIONS :  ',  predictions #ynn_test.iloc[:3].values,
    #)
    pred_test_val = [x['predictions'][0] for x in predictions]
    pred_train_val = [x['predictions'][0] for x in predictions_train]
    #print('pred_val', pred_val)
    return ynn_test.iloc[:].values, pred_test_val, ynn_train.values, pred_train_val

#######################################

# Xsnn_train, ynn_train, Xsnn_test, ynn_test  = gen_test_train(X_mag, t_mag)
# TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, LEARNING_RATE, TRAINING_EPOCHS, BATCH_SIZE, DISPLAY_STEP, N_HIDDEN_1 ,N_HIDDEN_2, N_INPUT, Nt_INPUT, Xsnn_train, Xsnn_test = initiate_nn(Xsnn_train, ynn_train, Xsnn_test, ynn_test)
#
# # if __name__ == "__main__":
# #     dft_test_data, pred_test_data, dft_train_data, pred_train_data = main()
#
# dft_test_data, pred_test_data, dft_train_data, pred_train_data = main()
