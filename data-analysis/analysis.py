'Exercise 1.1'
import random
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

random.seed(147)
df = pd.read_table('C:/Users/Ayesha Amin/PycharmProjects/projects/pca/Data/pca_toy.txt')
logging.info('reading the data')
print(df)

#######################################################################
#               a) Standardizing the variables                       #
#######################################################################
print()
logging.info(" Standardizing the variables")
features = ['a', 'b', 'c', 'd']  # (a) 'Standardization of pca_toy dataset'
x = df.loc[:, features].values  # Separating out the features
x = StandardScaler().fit_transform(x)
print("Standardization of pca_toy:: \n\n", x)

#######################################################################
#               b) Component extraction and plot                      #
#######################################################################
print()
logging.info('Perform PCA analysis')
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2'])
print(principalDf)

# Scatterplot formation of PC1 and PC2
print()
logging.info('Generating scatter plot')
targets = ['principal component 1', 'principal component 2']
fig = px.scatter_matrix(
    principalDf,
    dimensions=targets
)
fig.update_traces(diagonal_visible=False)
fig.show()
#plotly.offline.plot(fig, image='png', filename='image')

#######################################################################
#               c) Extracting Important Feature                       #
#######################################################################
print()
logging.info('Important Feature : Performing explanation')
n_pcs = pca.components_.shape[0]
# get the index of the most important feature on EACH component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = ['a', 'b', 'c', 'd']
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
output = pd.DataFrame(dic.items())
print(output)
print(abs(pca.components_))

#######################################################################
#               c) Variance Percentage of PC1 and PC2                 #
#######################################################################
print('')
logging.info('Variance')
exp_var_pca = (pca.explained_variance_ratio_)
print(exp_var_pca)

# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by principal component 1 and 2.
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
# Create the visualization plot

plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=1, align='center', label='Individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component 1 & 2')
plt.title('Percentage of Variance')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
