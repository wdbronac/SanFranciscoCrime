import theano.tensor as T
import lasagne
import onehot as oh
import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.basemap import Basemap
import csv
import pandas as pd
import sklearn
from sklearn.cluster import KMeans


#defining the parameters of the training
batch_size = 500
num_classes =39 
#TODO : definir un truc clean pour les inputs genre un truc qui fait bien des inputs, mais qui ne prend que la premiere partie et la met dans geo etc
#TODO : faire aussi un truc qui met les categories au bon format


def load_dataset():
    #import some points of the dataset
    print('Loading dataset...')
    df = pd.read_csv('../data/train.csv')
    print('Dataset loaded.')
    #convert the date into more precise inputs: 
    print('Converting...')
    month = oh.take_elems(df['Dates'],5,7)
    print('Month OK.')
    day= oh.take_elems(df['Dates'],8,10)
    print('Day OK.')
    year= oh.take_elems(df['Dates'],0,4)
    print('Year OK.')
    hour= oh.hm_to_seconds(df['Dates'],11)
    print('Hour OK.')
    longitudes = df['X'].as_matrix()
    print('Longitudes OK.')
    latitudes = df['Y'].as_matrix()
    print('Latitudes OK.')
    weekday = oh.one_hot(df['DayOfWeek']) # ca pourrait etre interessant de voir si on codait sous forme ordinale 
    print('DayOfWeek one-hot encoding OK.')
    district = oh.one_hot(df['PdDistrict']) # ca pourrait etre interessant de voir si on codait sous forme ordinale 
    print('District one-hot encoding OK.')
    #TODO : see if I add the district with a one hot encoding 
    #
    ## extract the label and convert it into numbers
    print('Transforming categories into indexes...')
    df['Category'] = pd.Series(df['Category'], dtype = 'category').cat.rename_categories(range(num_classes))
    y= df['Category'].values
    print('Categories transformed into indexes.')

    print('Creating the input vector...')
    X = latitudes.reshape(len(latitudes),1) #TODO : verifier que c est bien la bonne shape avec ipython
    print('Latitudes appended.')
    X = np.append(X, longitudes.reshape(len(longitudes),1), axis=1)
    print('Longitudes appended.')
    X = np.append(X,   month.reshape(len(month),1), axis=1)
    print('Month appended.')
    X = np.append(X,   day.reshape(len(day),1), axis=1)
    print('Day appended.')
    X = np.append(X,   year.reshape(len(year),1), axis=1)
    print('Year appended.')
    X = np.append(X,   hour.reshape(len(hour),1), axis=1)
    print('Hour appended.')
    X = np.append(X,   weekday, axis = 1)
    print('DayOfWeek appended.')
    X = np.append(X,   district, axis =1 )
    print('Disctrict appended.')

    print('Shuffling...')
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    # TODO : mettre les bons indices pour les inputs dans les couches
    X_current = X[idx] 
    y_current = y[idx] 

    print('Shuffled.')

    print('Dividing into training and test set...')
    X_train,X_val = X_current[:0.8*batch_size], X_current[0.8*batch_size:batch_size]
    y_train,y_val = y_current[:0.8*batch_size], y_current[0.8*batch_size:batch_size]
    print('Done.')
    return  X_train, y_train, X_val, y_val

def build_mlp(input_var = None): 
    #define the inputs of the different perceptrons
    input_geo = input_var[0:1] # input_geo: uniquement la latitude et la longitude
    input_geo_district = input_var[7]
    input_date = input_var[2:6]

    ##build the network connected to the geography input_geo > l_geo_hid2_drop
    #input layer
    l_geo_in = lasagne.layers.InputLayers(shape = input_geo.shape, input_var = input_geo)
    l_geo_in_drop = lasagne.layers.DropoutLayer(l_geo_in, p=0.2)
    #hidden layer
    l_geo_hid1 = lasagne.layers.DenseLayer(
            l_geo_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_geo_hid1_drop = lasagne.layers.DropoutLayer(l_geo_hid1, p=0.5)
    #hidden layer
    l_geo_hid2= lasagne.layers.DenseLayer(
            l_geo_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_geo_hid2_drop = lasagne.layers.DropoutLayer(l_geo_hid2, p=0.5)

    ##building the network for the district input_geo_district > l_geo_district_hid_2_drop
    #input layer
    l_geo_district_in = lasagne.layers.InputLayers(shape =  input_geo_district.shape, input_var = input_geo_district)
    l_geo_district_in_drop = lasagne.layers.DropoutLayer(l_in_geo_district, p=0.2)
    #hidden layer
    l_geo_district_hid1 = lasagne.layers.DenseLayer(
            l_geo_district_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_geo_district_hid1_drop = lasagne.layers.DropoutLayer(l_geo_district_hid1, p=0.5)
    #hidden layer
    l_geo_district_hid2= lasagne.layers.DenseLayer(
            l_geo_district_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_geo_district_hid2_drop = lasagne.layers.DropoutLayer(l_geo_district_hid2, p=0.5)


    #concatenating the layers of both geographic coordinates and district information: (l_geo_hid2_drop, l_date_hid2_drop) = l_tot_in
    l_geo_tot_in = lasagne.layers.ConcatLayer(l_geo_district_hid2_drop, l_geo_hid2_drop, axis = 1)

    #build the distributed representation for the geography: l_out_geo
    l_geo_tot_hid_1 = lasagne.layers.DenseLayer(l_geo_tot_in)
    l_geo_tot_hid1_drop = lasagne.layers.DenseLayer(
            l_geo_tot_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)



    #build the network connected to the date: l_date_in > l_date_hid2_drop
    l_date_in = lasagne.layers.InputLayers(shape =input_date.shape, input_var = input_date)
    l_date_in_drop = lasagne.layers.DropoutLayer(l_in_date, p=0.2)
    #hidden layer
    l_date_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_date_hid1_drop = lasagne.layers.DropoutLayer(l_date_hid1, p=0.5)
    #hidden layer
    l_date_hid2= lasagne.layers.DenseLayer(
            l_date_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_date_hid2_drop = lasagne.layers.DropoutLayer(l_date_hid2, p=0.5)


    #concatenating the layers of both geographic and time information: (l_geo_hid2_drop, l_date_hid2_drop) = l_tot_in
    l_tot_in = lasagne.layers.ConcatLayer(l_geo_hid2_drop, l_date_hid2_drop, axis = 1)

    #adding some layers for distributed representations: l_tot_in > l_common_hid2_drop
    l_common_in = lasagne.layers.InputLayers(shape = (None, None,1, len(input_common[0]) ), input_var = l_tot_in)
    l_common_in_drop = lasagne.layers.DropoutLayer(l_in_common, p=0.2)
    #hidden layer
    l_common_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_common_hid1_drop = lasagne.layers.DropoutLayer(l_common_hid1, p=0.5)
    #hidden layer
    l_common_hid2= lasagne.layers.DenseLayer(
            l_common_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_common_hid2_drop = lasagne.layers.DropoutLayer(l_common_hid2, p=0.5)

    #build the final layer: l_common_hid2drop > l_out: size: the number of classes (because one-hot encoding) 
    l_out = lasagne.layers.DenseLayer(
            l_common_hid2_drop, num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else :
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

def main():
    # Load the dataset
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train, y_train, X_val, y_val= load_dataset()
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    # Create neural network model
    network = build_mlp(input_var)

    prediction = lasagne.layers.get_output(l_out)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    rams = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)



    train_fn = theano.function([input_var, target_var], loss, updates=updates)


    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100))

