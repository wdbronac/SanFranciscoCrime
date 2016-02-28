import onehot as oh

#rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 10, max_depth = 5)

y= df['Category'].values

print('Converting data into standard input')
cols = [col for col in df.columns if col in [ 'Clus_loc']]
X = df[cols].values
print('Clus_loc...')
month = oh.take_elems(df['Dates'],5,7)
print('Months...')
day= oh.take_elems(df['Dates'],8,10)
print('Days...')
year= oh.take_elems(df['Dates'],0,4)
print('Years...')
hour= oh.hm_to_seconds(df['Dates'],11)
print('Hours...')

weekday = oh.one_hot(df['DayOfWeek']) # ca pourrait etre interessant de voir si on codait sous forme ordinale 
district = oh.one_hot(df['PdDistrict']) # ca pourrait etre interessant de voir si on codait sous forme ordinale 
print('WeekDay...')
print('Data converted')

X = np.append(X,   month.reshape(len(month),1), axis=1)
X = np.append(X,   day.reshape(len(day),1), axis=1)
X = np.append(X,   year.reshape(len(year),1), axis=1)
X = np.append(X,   hour.reshape(len(hour),1), axis=1)
X = np.append(X,   weekday, axis = 1)
X = np.append(X,   district, axis =1 )

#df_test = pd.read_csv('../data/test.csv', nrows = 100)

#X_test = df_test.ix[:, df_test.columns !='Category'].values
#y_test = df_test['Category'].values

y = df['Category'].values

print('Deleting intermediary objects...')
del(df)
del(month)
del(day)
del(year)
del(hour)
del(weekday)
del(clus_lat)
del(clus_long)
del(x_train)
del(longitudes)
del(latitudes)
del(district)
print('Objects deleted')
execfile('fit.py')











