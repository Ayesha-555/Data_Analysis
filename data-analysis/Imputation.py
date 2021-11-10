import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
print()
logging.info('===Performing data-analysis using basic statistical methods===')
print()
print()
logging.info('Reading the data')
dataset = pd.read_table('C:/Users/Ayesha Amin/PycharmProjects/projects/pca/Data/ms_toy.txt')
print(dataset)  # Displaying full data of 6960 rows x 6 columns
(dataset.info())  # Displaying the number of non-null values in each column
(dataset.isnull().sum())  # Calculate the number of missing values in each column

#######################################################################
#               1) Calculating mean & Standard Deviation              #
#######################################################################
logging.info('Calculating Mean & Standard Deviation')
print('Mean for the respective column \n\n', dataset.mean(), )
print('Standard Deviation for the respective column \n\n', dataset.std())

#######################################################################
#               2) Mean Value from Lower Quantile                     #
#######################################################################
print()
logging.info('Taking the average of the lower quantile distribution from the series and imputing the missing values')
l_quantile = dataset.quantile(0.25, interpolation="lower")
print(l_quantile)

avg = sum(l_quantile.tolist()) / len(l_quantile)
dataset.fillna(avg, inplace=True)
print(dataset)

#######################################################################
#               3) Standard Deviation                                 #
#######################################################################
print()
logging.info('New standard deviation by taking a fraction of the overall standard deviation')
new_std = l_quantile.std()
print(new_std)

#######################################################################
#               4) Randomly generating data                           #
#######################################################################
print()
logging.info('Generating new data by randomly drawing values from a normal distribution')
data = np.random.randn(1000)
print(data)

'Exercise 1.2 - 2'
# #######################################################################
# #               a) First three control samples                        #
# #######################################################################
print()
logging.info('Performing Imputation with KNN')
print()
logging.info('Extracting first three control samples from the data frame')
dataset = pd.read_table('C:/Users/Ayesha Amin/PycharmProjects/Data-Imputation/ms_toy.txt')
column_3 = dataset.iloc[:, :3]
print(column_3)

# #######################################################################
# #          b) Deleting all the rows containing missing values         #
# #######################################################################
print()
logging.info('Removing all the rows where the samples have missing values')
df = column_3.dropna(how='all')
print(df)#.to_string())


# #######################################################################
# #          c) KNN Imputation                                          #
# #######################################################################
print()
logging.info('KNN Imputation')
knn = KNNImputer(n_neighbors=3)
imputed = knn.fit_transform(df)
knn_imputed_df = pd.DataFrame(imputed, columns=['ctrl', 'ctrl2', 'ctrl3'])
print(knn_imputed_df)

# #######################################################################
# #          c) Histogram Plot                                          #
# #######################################################################
print()
logging.info('Histogram after KNN imutation')
imputed = knn_imputed_df.iloc[:, :1]
nan_list = [index for index, row in df.iterrows() if row.isnull().any()]
imputed_values = imputed[imputed.index.isin(nan_list)]  # Extracting all rows with  imputed values after KNN
temp = imputed_values.copy()
temp.columns = ['missing_values']
copy_imputed = imputed.copy()

# Assigning IDs to both columns
copy_imputed['ID'] = copy_imputed.index
temp['ID'] = temp.index
print(temp)
merged = pd.merge(copy_imputed, temp, left_on='ID', right_on='ID', how='left')
print(merged)#.to_string())

print("Extracting first (ctrl) and third (missing_values) column after merging")
both_col = merged[["ctrl", "missing_values"]]
print(both_col)#.to_string())

# Histogram
both_col.plot.hist(stacked=True, bins=20)
plt.xlabel('Intensity')
plt.ylabel('Count')
plt.legend(loc='best')
plt.show()


input("Press enter to exit :")