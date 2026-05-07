import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#data lode
train_df = pd.read_csv(r'D:\python code\DeepAIP\data\ESM特征\PIP-Train-ESM480..csv')
test_df  = pd.read_csv(r'D:\python code\DeepAIP\data\ESM特征\PIP-Test-ESM480..csv')
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)

#Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Applying PCA
pca = PCA(n_components=80)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
features_pca = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
#Creating new DataFrames for PCA-transformed data
train_pca_df = pd.DataFrame(X_train_pca, columns=features_pca)
train_pca_df.insert(0, 'label', y_train)
test_pca_df = pd.DataFrame(X_test_pca, columns=features_pca)
test_pca_df.insert(0, 'label', test_df['label'])
#Saving the transformed data
train_pca_df.to_csv(r'D:\python code\DeepAIP\data\ESM特征\PIP-Train-ESM80.csv', index=False)
test_pca_df.to_csv( r'D:\python code\DeepAIP\data\ESM特征\PIP-Test-ESM80.csv', index=False)
