#import csv from local disk
from google.colab import files
uploaded = files.upload()


#import modules
%matplotlib inline
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns


#OLS multiple regression with all variables

#df.dropna(inplace=True)
df.fillna(df.mean())

#set dependent TEA as exog
Y = df[df.columns[1]]

#set all other variables as endog
X = df[df.columns[2:]]

#add constant
X = sm.add_constant(X)
model = sm.OLS(Y, X)
  
results = model.fit()
print(results.summary())


#VIF
col_num = model.exog.shape[1] 
vifs = [variance_inflation_factor(model.exog, i) for i in range(0, col_num)]
pd.DataFrame(vifs, index=model.exog_names, columns=["VIF"])


# linearlity
def plot_scatter(endog):
  
  exog = df.columns[1]
  x = df[endog]
  Y = df[exog]

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.scatter(x, Y)
  
  ax.set_title('testify linearlity')
  ax.set_xlabel(endog)
  ax.set_ylabel(exog)

  fig.show()

# get length of lines
length_df = len(df.columns)

# testify linearlity
for i in range(2, length_df):
  
  endog = df.columns[i]
  plot_scatter(endog)
  
  i += 1


# homoscedasticity
breuschpagan_labal = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']


# breuschpagan test
breuschpagan_results = sm.stats.diagnostic.het_breuschpagan(results.resid, results.model.exog)

for name, result in zip(breuschpagan_labal, breuschpagan_results):
    print (name, result)

data = pd.read_excel('Variables.xlsx', encoding='utf-8', header=0)
pd.set_option('display.expand_frame_repr', False)


# calculate total effect
def calc_total_effect(column_name, number):
  
  aquire constant
  const = results.params[0]
 
  #aquire length of data
  length_df = len(data)
  
  variable = data[column_name]
  
  #set an array for total_effect
  array_total_effect = []
  
  for i in range(0, length_df):
    
    total_effect = (results.params[number] * variable[i])
    array_total_effect.append(total_effect)
  
  #convert Dataframe to an array
  x = np.array(data['Country'])
  y = np.array(array_total_effect)

  x_position = np.arange(len(x))
  plt.figure(figsize=(200, 5), dpi = 90)
  plt.bar(x_position, y, tick_label=x )

  #shape figures
  plt.title(column_name)
  plt.xlabel('country code')
  plt.ylabel('Total effect of\n' + column_name)
  

calc_total_effect('Hoge', 1)