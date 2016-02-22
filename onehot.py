import numpy as np
def one_hot(df):
  """
  Fonction qui transforme un dataserie en one hot encoding. Pourrait surement etre optimise.
  """
  vec = df.unique()
  total = np.zeros((df.shape[0],len(vec)))
  for i in range(df.size):
    index = np.where(vec == df[i])
    total[i,index] = 1
  return total


def take_elems(df, a, b):
  """
  df is a data serie of dates, so that we can take only some parts of the date, between a and b (for instance only the hour)
  """
  total = np.zeros(df.shape[0])
  for i in range(df.shape[0]) :
    total[i]  = float(df[i][a:b])
  return total



def hm_to_seconds(df, a):
  """
  this is more or less the same than take_elems, but what is more it converts an hour into a count of minutes here, no b because the hour is at the end of the array
  """

  total = np.zeros(df.shape[0])
  for i in range(df.shape[0]): 
    t = df[i][a:]
    h, m,s = [int(k) for k in t.split(':')]
    total[i]  = float(3600*h + 60*m + s)
  return total


