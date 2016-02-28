from sklearn.ensemble import RandomForestClassifier
rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 50 )
idx = np.arange(X.shape[0])
#np.random.seed(RANDOM_SEED)
#creating batches and fitting
print('Pre-processing finished')
print('Creating batches.')
#batch_size = 50000
batch_size = len(X)
num_batch = X.shape[0]/batch_size 
np.random.shuffle(idx)
print('Fitting : ')
for i in range(num_batch):
  batch_idx = idx[i * batch_size: (i+1) * batch_size]
  print "Batch %d of %d :"% (i , num_batch)
  X_current = X[batch_idx]
  y_current = y[batch_idx]
  X_train,X_test = X_current[:0.8*batch_size], X_current[0.8*batch_size:batch_size]
  y_train,y_test = y_current[:0.8*batch_size], y_current[0.8*batch_size:batch_size]
  print('Fitting...')
  rf.fit(X_train,y_train)
  print('Fitted.')
  score = rf.score(X_test, y_test)
  print "Score: %f" % score
  #computes the logloss on this batch
  proba = rf.predict_log_proba(X_test)
  
  score_log = sum(proba[_,y_test[_]] for _ in range(len(y_test)))
  print "Score_log: %f" % score_log
