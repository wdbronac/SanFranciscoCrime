import cPickle
import theano
import theano.tensor as T
import lasagne
import time
import onehot as oh
import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.basemap import Basemap
import csv
import pandas as pd


#defining the parameters of the training
batch_size = 20000
num_classes =39 
size_one_hot_district = 4
num_epochs = 60
#TODO : definir un truc clean pour les inputs genre un truc qui fait bien des inputs, mais qui ne prend que la premiere partie et la met dans geo etc
#TODO : faire aussi un truc qui met les categories au bon format


def load_dataset( reload = False, test = False):
    if test == True: 
	filename = '../data/test.csv'
    else:
	filename = '../data/train.csv'

    if reload == True:
        #import some points of the dataset
        print('Loading dataset...')
        df = pd.read_csv(filename)#convert the date into more precise inputs: 
	if test == False:
		categories= np.array(pd.Series(df['Category'], dtype = 'category').cat.categories)
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
	if test == False: 
		df['Category'] = pd.Series(df['Category'], dtype = 'category').cat.rename_categories(range(num_classes))
		y= df['Category'].values
		print('Categories transformed into indexes.')
                classes = pd.Series(df['Category'], dtype = 'category').cat.categories    
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
        X_max = X.max(axis=0)
        if test == False:
            X = X/X_max
	if test == True: 
		f = open('X_max.save', 'rb')
		X_max= cPickle.load(f)
		f.close()
		print('Saving dataset...')
		f = open('X_test.save', 'wb')
                X = X/X_max
		cPickle.dump(X, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		return X
        else: 
		print('Shuffling...')
		idx = np.arange(X.shape[0])
		np.random.shuffle(idx)
		# TODO : mettre les bons indices pour les inputs dans les couches
		X_current = X[idx] 
		y_current = y[idx] 

		#converting y into nparray
		y_current = np.array(y_current)

		print('Shuffled.')

		print('Dividing into training and test set...')
		X_train,X_val = X_current[:0.8*len(X_current)], X_current[0.8*len(X_current):len(X_current)]
		y_train,y_val = y_current[:0.8*len(y_current)], y_current[0.8*len(y_current):len(y_current)]
		print('Done.')

       # f = open('data.save', 'wb')
       # for obj in [X_train, y_train, X_val, y_val]:
       #     cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
       # f.close()

		print('Saving dataset...')
		f = open('X_train.save', 'wb')
		cPickle.dump(X_train, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		f = open('y_train.save', 'wb')
		cPickle.dump(y_train, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		f = open('X_val.save', 'wb')
		cPickle.dump(X_val, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		f = open('y_val.save', 'wb')
		cPickle.dump(y_val, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		f = open('classes.save', 'wb')
		cPickle.dump(classes, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		f = open('categories.save', 'wb')
		cPickle.dump(categories, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		f = open('X_max.save', 'wb')
		cPickle.dump(X_max, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		print('Dataset saved.')
		return  X_train, y_train, X_val, y_val, classes, categories, X_max
    else : 
        #f = open('data.save', 'r')
        #loaded_objects = []
        #for i in range(4):
        #    loaded_objects.append(cPickle.load(f))
        #f.close()
	if test == True: 
		print('Loading dataset...')
		f = open('X_test.save', 'rb')
		X_test = cPickle.load(f)
		f.close()
		return X_test
	else:
		print('Loading dataset...')
		f = open('X_train.save', 'rb')
		X_train = cPickle.load(f)
		f.close()
		f = open('y_train.save', 'rb')
		y_train = cPickle.load(f)
		f.close()
		f = open('X_val.save', 'rb')
		X_val = cPickle.load(f)
		f.close()
		f = open('y_val.save', 'rb')
		y_val= cPickle.load(f)
		f.close()
		f = open('classes.save', 'rb')
		classes= cPickle.load(f)
		f.close()
		f = open('categories.save', 'rb')
		categories= cPickle.load(f)
		f.close()
		f = open('X_max.save', 'rb')
		X_max= cPickle.load(f)
		f.close()
		print('Dataset loaded.')
        #X_train, y_train, X_val, y_val = loaded_objects
		return  X_train, y_train, X_val, y_val, classes, categories, X_max
    return 0


def build_mlp(input_var = None): 
    print('Building model...')
    #define the inputs of the different perceptrons
    #input_geo = input_var[0:1] # input_geo: uniquement la latitude et la longitude
    #input_geo_district = input_var[9:12]
    #input_date = input_var[2:8]
    l_in= lasagne.layers.InputLayer(shape = (None,23), input_var = input_var)
    ##build the network connected to the geography input_geo > l_geo_hid2_drop
    #input layer
    l_geo_in = lasagne.layers.SliceLayer(l_in, indices=slice(0, 1), axis=1)
    #l_geo_in = lasagne.layers.InputLayer(shape = input_geo.shape, input_var = input_geo)
    #l_geo_in_drop = lasagne.layers.DropoutLayer(l_geo_in, p=0.2)
    l_geo_in_drop = l_geo_in
    #hidden layer
    l_geo_hid1 = lasagne.layers.DenseLayer(
            l_geo_in_drop, num_units=900,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_geo_hid1_drop = lasagne.layers.DropoutLayer(l_geo_hid1, p=0.5)
    #hidden layer
    l_geo_hid2= lasagne.layers.DenseLayer(
            l_geo_hid1_drop, num_units=900,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_geo_hid2_drop = lasagne.layers.DropoutLayer(l_geo_hid2, p=0.5)

    l_geo_hid3= lasagne.layers.DenseLayer(
            l_geo_hid2_drop, num_units=900,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_out_geo = lasagne.layers.DropoutLayer(l_geo_hid3, p=0.5)

    ##building the network for the district input_geo_district > l_geo_district_hid_2_drop
    #input layer
    #input_geo_district = lasagne.layers.SliceLayer(l_in, indices=slice(9,12))
    output_geo_district = lasagne.layers.SliceLayer(l_in, indices=slice(12,22), axis=1)
    #l_geo_district_in = lasagne.layers.InputLayer(shape =  input_geo_district.shape, input_var = input_geo_district)
    #l_geo_district_in_drop = lasagne.layers.DropoutLayer(l_geo_district_in, p=0.2)
    #hidden layer
#    l_geo_district_hid1 = lasagne.layers.DenseLayer(
#                   # l_geo_district_in_drop, num_units=802,
#                           input_geo_district, num_units=10,
#                                   nonlinearity=lasagne.nonlinearities.rectify,
#                                           W=lasagne.init.GlorotUniform())
#    l_geo_district_hid1_drop = lasagne.layers.DropoutLayer(l_geo_district_hid1, p=0.5)
#    #hidden layer
#    l_geo_district_hid2= lasagne.layers.DenseLayer(
#                    l_geo_district_hid1_drop, num_units=10,
#                            nonlinearity=lasagne.nonlinearities.rectify)
#    l_geo_district_hid2_drop = lasagne.layers.DropoutLayer(l_geo_district_hid2, p=0.5)
#

    #concatenating the layers of both geographic coordinates and district information: (l_geo_hid2_drop, l_date_hid2_drop) = l_tot_in
    l_geo_tot_in = lasagne.layers.concat([output_geo_district, l_out_geo], axis=1)
    #l_geo_tot_in_drop = lasagne.layers.DropoutLayer(l_geo_tot_in, p=0.2)
    l_geo_tot_out = l_geo_tot_in

    #build the distributed representation for the geography: l_out_geo
    l_geo_tot_hid1 = lasagne.layers.DenseLayer(
            l_geo_tot_out, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_geo_tot_hid_1_drop = lasagne.layers.DropoutLayer(l_geo_tot_hid1, p=0.5)
    l_geo_tot_hid2 = lasagne.layers.DenseLayer(
            l_geo_tot_hid_1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_geo_tot_out = lasagne.layers.DropoutLayer(l_geo_tot_hid2, p=0.5)


    #TODO : attention j ai ecrit exactement la meme chose ici
    #build the network connected to the date: l_date_in > l_date_hid2_drop
    l_date_in = lasagne.layers.SliceLayer(l_in, indices=slice(2,12), axis=1)
    #l_date_in_drop = lasagne.layers.DropoutLayer(l_date_in, p=0.2)
    #hidden layer
    l_date_hid1 = lasagne.layers.DenseLayer(
            l_date_in, num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_date_hid1_drop = lasagne.layers.DropoutLayer(l_date_hid1, p=0.5)
    #hidden layer
    l_date_hid2= lasagne.layers.DenseLayer(
            l_date_hid1_drop, num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_date_out = lasagne.layers.DropoutLayer(l_date_hid2, p=0.5)


    #concatenating the layers of both geographic and time information: (l_geo_hid2_drop, l_date_hid2_drop) = l_tot_in
    l_tot_in = lasagne.layers.concat([l_geo_tot_out, l_date_out], axis=1)
    l_tot_in_drop = lasagne.layers.DropoutLayer(l_tot_in, p=0.2)

    #adding some layers for distributed representations: l_tot_in > l_common_hid2_drop
    l_common_in = lasagne.layers.DenseLayer(
            l_tot_in_drop, num_units=50*2,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_common_in_drop = lasagne.layers.DropoutLayer(l_common_in, p=0.2)
    #hidden layer
    l_common_hid1 = lasagne.layers.DenseLayer(
            l_common_in_drop, num_units=70,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_common_hid1_drop = lasagne.layers.DropoutLayer(l_common_hid1, p=0.5)
    #hidden layer
    l_common_hid2= lasagne.layers.DenseLayer(
            l_common_hid1_drop, num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_common_hid2_drop = lasagne.layers.DropoutLayer(l_common_hid2, p=0.5)

    #build the final layer: l_common_hid2drop > l_out: size: the number of classes (because one-hot encoding) 
    l_out = lasagne.layers.DenseLayer(
            l_common_hid2_drop, num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax)
    print('Model built.')
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

def main(debug = False):
    # Load the dataset
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train, y_train, X_val, y_val, classes, categories= load_dataset()
    # Prepare Theano variables for inputs and targets
    if debug == False: 
        input_var = T.fmatrix('inputs')
        target_var = T.ivector('targets')
        # Create neural network model
        network = build_mlp(input_var)

        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=0.01, momentum=0.9)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)



        train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)


        val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            print 'Training:'
            idxloc = 0;
            for  batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                #faire une barre de chargement
                idxloc += 1;
                #if(idxloc%(len(X_train)/batch_size/10)==0):
                    #print idxloc*100/(len(X_train)/batch_size), '%'
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1
                # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            print('Testing:')
            idxloc = 0;
            for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
                idxloc += 1;
                #if(idxloc%(len(X_val)/batch_size/10)==0):
                    #print idxloc*100/(len(X_val)/batch_size), '%'
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
            print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
#            # Then we print the results for this epoch:
#            print("Epoch {} of {} took {:.3f}s".format(
#            epoch + 1, num_epochs, time.time() - start_time))
#            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
#            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
#            print("  validation accuracy:\t\t{:.2f} %".format(
#                val_acc / val_batches * 100))
    else : 
        def predict_function(X): 
            res = np.random.randn(len(X), 39)
            return res
    # Then we print the results for this epoch:

    if debug ==False:
        prediction = lasagne.layers.get_output(network, deterministic=True)
        predict_function = theano.function([input_var], prediction,allow_input_downcast=True )

    X_test = load_dataset(  test = True)	
    pred = np.empty((0, 39))
    #result =  predict_function(X_test)
    #df_final = pd.DataFrame(result, columns=["Id", classes])
    #df_final.to_csv('../data/finalSubmission2.csv')  
    del(X_train)
    del(X_val)
    del(y_train)
    del(y_val)
    file=open("../data/submit.csv", "wb") 
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(np.append("Id", categories))
    pred = np.empty((0,39))
    memsize = 100000
    print('Writing the data...')
    index = 0
    for start_idx in range(0, len(X_test) - memsize + 1, memsize):
        prediction = predict_function(X_test[start_idx:start_idx+memsize])
        for row_current in prediction: 
            writer.writerow(np.append(index, row_current))
            index +=1;
            #if index*100%(len(X_test))==0:
                #print index*100/(len(X_test)), '%'
    prediction = predict_function(X_test[-(len(X_test)%memsize):])
    for row_current in prediction: 
        writer.writerow(np.append(index, row_current))
        index +=1;
        #if index%(len(X_test))==0:
            #print index*100/(len(X_test)), '%'
    del(prediction)
    file.close()
    print('Data written.')
    

