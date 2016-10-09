#!/usr/bin/env python

#from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import sys
import lasagne

window_width = 129

class CustomDenseLayer(lasagne.layers.DenseLayer):
    
    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
            
        output = self.nonlinearity(activation)
        #print (output)
        return output.reshape((output.shape[0],1,output.shape[1]))
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1,self.num_units)
    
def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    

    target_app = 'kettle'
    window_width = 129
    
    X_train = np.load('E:/dataset/'+ target_app +'/task_3/'+target_app+'_tra_x.npy')#[0:100000]
    y_train = np.load('E:/dataset/'+ target_app +'/task_3/'+target_app+'_tra_y.npy')#[0:100000]#.reshape(X_train.shape[0],1)#.reshape((Xy_train.shape[0],1,1,Xy_train.shape[1]/2))
    #y_train = y_train.astype('uint8')
    
    X_val = np.load('E:/dataset/'+ target_app +'/task_3/'+target_app+'_val_x.npy')#[0:100]
    y_val = np.load('E:/dataset/'+ target_app +'/task_3/'+target_app+'_val_y.npy')#[0:100]#.reshape(X_val.shape[0],1)#.reshape((Xy_val.shape[0],1,1,Xy_val.shape[1]/2))
    #y_val = y_val.astype('uint8')
    
    X_test = np.load('E:/dataset/'+ target_app +'/task_3/'+target_app+'_test_x.npy')#[0:10000]
    y_test = np.load('E:/dataset/'+ target_app +'/task_3/'+target_app+'_test_y.npy')#[0:10000]#.reshape(X_test.shape[0],1)#.reshape((Xy_test.shape[0],1,1,Xy_test.shape[1]/2))
    #y_test = y_test.astype('uint8')
    
    print('X_train', X_train.shape, y_train.shape)
    print('X_val', X_val.shape, y_val.shape)
    print('X_test', X_test.shape, y_test.shape)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.



def build_test(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, window_width), input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=window_width*2, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=window_width*4)
    l_out = lasagne.layers.DenseLayer(l_hid2, num_units=256, nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, window_width), input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=window_width*4, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=window_width*6)
    l_hid3 = lasagne.layers.DenseLayer(l_hid1, num_units=window_width*8)
    l_out = lasagne.layers.DenseLayer(l_hid2, num_units=256, nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


def build_cnn(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, window_width), input_var=input_var)
    l_hid1 = lasagne.layers.Conv1DLayer(l_in, num_filters= 10 , filter_size = 8, stride = 1, pad = 'valid', nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform()) 
    l_hid2 = lasagne.layers.Conv1DLayer(l_hid1, num_filters= 6 , filter_size = 4, stride = 1, pad = 'valid', nonlinearity=None, W=lasagne.init.GlorotUniform())
    l_hid3 = lasagne.layers.DenseLayer(l_hid2, num_units=window_width*6, nonlinearity=lasagne.nonlinearities.sigmoid)
    l_hid4 = lasagne.layers.DenseLayer(l_hid3, num_units=window_width*8, nonlinearity=lasagne.nonlinearities.rectify)
    l_out = lasagne.layers.DenseLayer(l_hid4, num_units=256, nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


def get_output(data,fragment, get):
    
    num = data.shape[0]/fragment
    offset = data.shape[0]%fragment
    final_output = np.zeros((data.shape[0]))
    
    for i in range(num):
        #print "processing data", i+1
        output = get(data[i*fragment:(i+1)*fragment])
        final_output[i*fragment:(i+1)*fragment] = output
    
    #print 'offset:', offset
    output = get(data[-offset:])
    final_output[-offset:] = output
    
    return final_output


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='mlp', num_epochs=500, batch_size = 500):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')
    
    
    lr = 0.01
    tra_loss = np.zeros((num_epochs))
    val_loss = np.zeros((num_epochs))
    
    
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
   
    network = build_mlp(input_var)
       
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.nesterov_momentum(
            #loss, params, learning_rate=lr, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    get_prediction = T.argmax(test_prediction, axis=1)
    # As a bonus, also create an expression for the classification accuracy:
#     test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
#                       dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    #train_fn = theano.function([input_var, target_var], [test_loss, test_acc], updates=updates, allow_input_downcast = True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast = True)
    
    output = theano.function([input_var], get_prediction, allow_input_downcast = True)
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        decay = (epoch/5+1)
        print 'learning rate:', lr/decay
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=lr/decay, momentum=0.9)
        train_fn = theano.function([input_var, target_var], [test_loss, test_acc], updates=updates, allow_input_downcast = True)
        
        train_err = 0
        train_acc = 0 
        train_batches = 0
        start_time = time.time()
        batch_num = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            batch_num += 1
            inputs, targets = batch
            #print('here, ', inputs[0].shape, targets[1].shape)
            err,acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            #print batch_num, train_err
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 1000, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(
             train_acc / train_batches * 100))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        #print("  validation accuracy:\t\t{:.6f}".format(val_acc / val_batches))
        tra_loss[epoch] = train_err / train_batches
        val_loss[epoch] = val_err / val_batches
        print("  validation accuracy:\t\t{:.2f} %".format(
             val_acc / val_batches * 100))
        sys.stdout.flush()
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 1000, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
         test_acc / test_batches * 100))
    test_loss = test_err / test_batches

    layers = lasagne.layers.get_all_param_values(network)
    
    network_analysis = [ ]
    network_analysis.append(layers)
    network_analysis.append(tra_loss)
    network_analysis.append(val_loss)
    network_analysis.append(test_loss)
    
    pre_out = get_output(X_test,10000, output)

        
    
    return network_analysis, pre_out


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = 'mlp'
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = 20
        network_analysis, output= main(**kwargs)
