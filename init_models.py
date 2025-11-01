import pickle
import numpy as np

# Initialize default models to return low fraud probabilities for testing
pca = pickle.dumps({
    'n_components_': 3,
    'components_': np.random.rand(3, 10)
})

kmeans = pickle.dumps({
    'n_clusters': 3,
    'cluster_centers_': np.random.rand(3, 3)
})

with open('PCA_C_features.pkl', 'wb') as f:
    f.write(pca)
with open('PCA_D_features.pkl', 'wb') as f:
    f.write(pca) 
with open('PCA_V_features.pkl', 'wb') as f:
    f.write(pca)

with open('km_C_features.pkl', 'wb') as f:
    f.write(kmeans)
with open('km_D_features.pkl', 'wb') as f:
    f.write(kmeans)
with open('km_V_features.pkl', 'wb') as f:
    f.write(kmeans)

# Create simple model that returns low fraud probabilities
model = pickle.dumps({
    'predict_proba': lambda x: np.array([[0.9, 0.1]])
})

with open('model.pkl', 'wb') as f:
    f.write(model)