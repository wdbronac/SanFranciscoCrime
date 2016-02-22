#from theano import lasagne

import onehot as oh
import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.basemap import Basemap
import csv
import pandas as pd
import sklearn
from sklearn.cluster import KMeans

#import some points of the dataset
df = pd.read_csv('../data/train.csv', nrows = 1000)

#convert the date into more precise inputs: 
month = oh.take_elems(df['Dates'],5,7)
day= oh.take_elems(df['Dates'],8,10)
year= oh.take_elems(df['Dates'],0,4)
hour= oh.hm_to_seconds(df['Dates'],11)
longitudes = df['X'].as_matrix()
latitudes = df['Y'].as_matrix()
weekday = oh.one_hot(df['DayOfWeek']) # ca pourrait etre interessant de voir si on codait sous forme ordinale 
#TODO : see if I add the district with a one hot encoding 

#
## extract the label and convert it into one hot
category = oh.one_hot(df['Category'])
#
#
#
#
#
## remove the useless data so as to keep only the useful data in a numpy array
#
#
#
#
#
#
##build the neural net

def build_mlp(input_var = None): 

    ##build the network connected to the geography
    #input layer
    l_geo_in = lasagne.layers.InputLayers(shape = (None,2), input_var = input_var)
    l_geo_in_drop = lasagne.layers.DropoutLayer(l_in_geo, p=0.2)
    #hidden layer
    l_geo_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_geo_hid1_drop = lasagne.layers.DropoutLayer(l_geo_hid1, p=0.5)
    #hidden layer
    l_geo_hid2= lasagne.layers.DenseLayer(
            l_geo_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    l__geo_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    ##building the network for the district
    #input layer
    l_geo_in = lasagne.layers.InputLayers(shape = (None,2), input_var = input_var)
    l_geo_in_drop = lasagne.layers.DropoutLayer(l_in_geo, p=0.2)
    #hidden layer
    l_geo_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_geo_hid1_drop = lasagne.layers.DropoutLayer(l_geo_hid1, p=0.5)
    #hidden layer
    l_geo_hid2= lasagne.layers.DenseLayer(
            l_geo_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    l__geo_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    #build the distributed representation for the geography: l_out_geo
    l_tot_hid_1 = lasagne.layers.DenseLayer(l_tot_in)
    l_tot_hid1_drop = lasagne.layers.DenseLayer(
            l_tot_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    #build the network connected to the date: l_out_date
    l_in_date = lasagne.layers.InputLayers(shape = (None,2), input_var = input_var)
    l_in_date_drop = lasagne.layers.DropoutLayer(l_in_date, p=0.2)
    l_in_date_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    #concatenating the layers of both geographic and time information
    l_tot_in = lasagne.layers.ConcatLayer(l_out_geo, l_out_date, axis = 1)

    #adding some layers for distributed representations
    l_tot_hid_1 = lasagne.layers.DenseLayer(l_tot_in)
    l_tot_hid1_drop = lasagne.layers.DenseLayer(
            l_tot_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)


    #build the final layer 
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
