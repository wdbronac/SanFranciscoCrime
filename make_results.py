
#import some points of the dataset
print('Loading the test data...')
df_test = pd.read_csv('../data/test.csv')
print('Data Loaded.')
#df_test = pd.read_csv('../data/train.csv', nrows = 50000)

#LENGTH = 100000
LENGTH_test =df_test.size 
print('Pre-processing...\n')
longitudes = df_test['X'].as_matrix()
latitudes = df_test['Y'].as_matrix()
# replacing categories by numbers labels
#computes the prototypes of kmeans and add it to latitude and longitude
print('Geographic clustering...: giving the right cluster to the coordinates')
x_test = df_test[['X','Y']];
#create the cluster points and add it to the data
new_X_test = clus.predict(x_test)
df_test['Clus_loc']=new_X_test
print('Clustering finished.')


print('Converting data into standard input')
cols_test = [col for col in df_test.columns if col in [ 'Clus_loc']]
X_test = df_test[cols_test].values
print('Clus_loc...')
month_test = oh.take_elems(df_test['Dates'],5,7)
print('Months...')
day_test= oh.take_elems(df_test['Dates'],8,10)
print('Days...')
year_test= oh.take_elems(df_test['Dates'],0,4)
print('Years...')
hour_test= oh.hm_to_seconds(df_test['Dates'],11)
print('Hours...')

weekday_test = oh.one_hot(df_test['DayOfWeek']) # ca pourrait etre interessant de voir si on codait sous forme ordinale 
district_test = oh.one_hot(df_test['PdDistrict']) # ca pourrait etre interessant de voir si on codait sous forme ordinale 
print('WeekDay...')
print('Data converted')

X_test = np.append(X_test,   month_test.reshape(len(month_test),1), axis=1)
X_test = np.append(X_test,   day_test.reshape(len(day_test),1), axis=1)
X_test = np.append(X_test,   year_test.reshape(len(year_test),1), axis=1)
X_test = np.append(X_test,   hour_test.reshape(len(hour_test),1), axis=1)
X_test = np.append(X_test,   weekday_test, axis = 1)
X_test = np.append(X_test,   district_test, axis =1 )

print('Processing finished')
print('Starting prediction')


prediction = rf.predict_proba(X_test)
df_final = pd.DataFrame(prediction, columns=["Id", classes])

df_final.to_csv('../data/finalSubmission2.csv')

