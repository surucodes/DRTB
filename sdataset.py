import pandas as pd
from io import StringIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# The original dataset provided as a string
data_str = """Gender	Age	Contact DR	Smoking	Alcohol	Cavitary pulmonary	Diabetes	Nutritional	TBoutside	Class
Male	>= 45 years	Yes	Yes	No	Yes	No	Underweight	No	DR
Female	>= 45 years	Yes	Yes	No	Yes	No	Underweight	No	DR
Female	>= 45 years	No	No	No	Yes	No	Underweight	No	DR
Male	>= 45 years	Yes	Yes	No	Yes	No	Normal	No	DR
Female	< 45 years	No	No	No	Yes	No	Underweight	No	DR
Male	>= 45 years	Yes	Yes	No	Yes	No	Underweight	No	DR
Male	< 45 years	No	Yes	No	Yes	No	Normal	No	DR
Male	>= 45 years	Yes	Yes	No	Yes	No	Underweight	No	DR
Male	< 45 years	Yes	Yes	No	Yes	No	Underweight	No	DR
Male	>= 45 years	No	Yes	Yes	Yes	No	Underweight	No	DR
Female	< 45 years	Yes	No	No	Yes	No	Underweight	No	DR
Male	>= 45 years	Yes	Yes	No	Yes	No	Underweight	No	DR
Female	>= 45 years	Yes	No	No	Yes	No	Underweight	Yes	DR
Male	< 45 years	Yes	Yes	No	Yes	No	Underweight	No	DR
Female	< 45 years	No	No	No	Yes	No	Normal	No	DR
Male	>= 45 years	Yes	Yes	No	Yes	No	Underweight	No	DR
Female	>= 45 years	Yes	Yes	No	Yes	No	Normal	No	DR
Male	< 45 years	Yes	Yes	No	Yes	Yes	Underweight	No	DR
Male	>= 45 years	No	Yes	No	Yes	No	Normal	No	DR
Male	>= 45 years	Yes	Yes	No	Yes	Yes	Underweight	No	DR
Female	>= 45 years	Yes	No	No	Yes	No	Normal	No	DR
Female	>= 45 years	Yes	No	No	Yes	No	Normal	No	DR
Female	>= 45 years	Yes	Yes	No	Yes	No	Normal	No	DR
Male	< 45 years	Yes	Yes	No	Yes	No	Underweight	No	DR
Male	>= 45 years	Yes	Yes	Yes	No	No	Underweight	No	DR
Male	>= 45 years	Yes	Yes	Yes	No	Yes	Normal	No	DR
Female	< 45 years	Yes	No	No	No	No	Normal	Yes	DR
Female	>= 45 years	Yes	No	No	No	Yes	Normal	No	DR
Male	>= 45 years	Yes	Yes	Yes	No	No	Underweight	No	DR
Female	>= 45 years	Yes	No	No	No	No	Underweight	No	DR
Male	>= 45 years	Yes	Yes	No	No	Yes	Underweight	No	DR
Male	>= 45 years	Yes	No	No	No	No	Underweight	No	DR
Male	>= 45 years	Yes	Yes	No	No	Yes	Underweight	No	DR
Male	>= 45 years	Yes	No	No	No	No	Underweight	Yes	DR
Male	< 45 years	Yes	Yes	No	No	No	Underweight	No	DR
Male	< 45 years	Yes	No	No	No	No	Underweight	No	DR
Female	>= 45 years	Yes	No	No	No	Yes	Underweight	No	DR
Male	>= 45 years	Yes	Yes	No	No	Yes	Normal	No	DR
Female	>= 45 years	Yes	No	No	No	No	Normal	No	DR
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DR
Male	< 45 years	No	No	No	No	No	Normal	No	DR
Male	>= 45 years	Yes	Yes	No	No	No	Underweight	No	DR
Male	>= 45 years	No	No	No	No	Yes	Normal	No	DR
Female	< 45 years	Yes	No	No	No	No	Underweight	No	DR
Male	>= 45 years	Yes	No	No	No	No	Underweight	No	DR
Male	>= 45 years	Yes	Yes	No	Yes	Yes	Underweight	No	DR
Female	>= 45 years	Yes	Yes	No	No	No	Underweight	No	DR
Male	>= 45 years	Yes	Yes	No	No	No	Underweight	No	DR
Female	>= 45 years	Yes	No	No	No	No	Underweight	No	DR
Male	>= 45 years	No	Yes	No	No	No	Normal	Yes	DR
Male	>= 45 years	Yes	Yes	Yes	No	Yes	Underweight	No	DR
Male	>= 45 years	No	No	No	No	No	Normal	Yes	DR
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DR
Male	< 45 years	No	No	No	No	No	Normal	No	DR
Male	< 45 years	Yes	No	No	No	No	Underweight	Yes	DR
Female	< 45 years	Yes	No	No	No	No	Normal	No	DR
Female	< 45 years	Yes	No	No	No	No	Underweight	No	DR
Male	< 45 years	Yes	No	No	No	No	Underweight	No	DR
Male	>= 45 years	Yes	Yes	Yes	No	No	Underweight	No	DR
Male	>= 45 years	Yes	No	No	No	Yes	Underweight	No	DR
Male	>= 45 years	Yes	No	No	No	Yes	Normal	Yes	DR
Male	>= 45 years	Yes	No	No	No	No	Normal	Yes	DR
Male	>= 45 years	No	No	No	No	Yes	Normal	No	DR
Male	>= 45 years	No	No	No	No	Yes	Normal	No	DR
Female	< 45 years	No	No	No	No	Yes	Normal	No	DR
Female	>= 45 years	Yes	No	No	No	Yes	Normal	Yes	DR
Female	>= 45 years	No	No	No	No	Yes	Normal	Yes	DR
Female	>= 45 years	Yes	No	No	No	Yes	Normal	Yes	DR
Female	< 45 years	Yes	No	No	No	No	Underweight	No	DR
Female	>= 45 years	No	No	No	No	Yes	Normal	No	DR
Male	>= 45 years	Yes	Yes	No	No	No	Underweight	No	DR
Male	< 45 years	No	No	No	No	Yes	Normal	No	DR
Male	>= 45 years	No	Yes	No	No	Yes	Normal	No	DR
Male	< 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	Yes	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	Yes	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	< 45 years	No	Yes	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	Yes	Normal	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	No	Normal	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	Yes	Normal	No	DS
Male	< 45 years	No	Yes	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	< 45 years	No	Yes	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	< 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	Yes	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	Yes	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	No	No	No	Yes	Normal	No	DS
Male	< 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	< 45 years	Yes	No	No	No	No	Underweight	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	No	No	No	Yes	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	< 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Underweight	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	Yes	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	Yes	Normal	No	DS
Male	< 45 years	Yes	Yes	Yes	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	Yes	Underweight	No	DS
Female	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	Yes	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	No	No	No	Yes	Normal	No	DS
Female	< 45 years	No	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Female	< 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	No	No	No	No	Underweight	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	< 45 years	Yes	No	No	No	No	Underweight	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	< 45 years	Yes	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	Yes	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	Yes	Underweight	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	Yes	Normal	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	Yes	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Male	< 45 years	No	Yes	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	Yes	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	Yes	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	Yes	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	Yes	Underweight	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	No	Yes	No	No	No	Underweight	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	No	No	No	Yes	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	< 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Female	>= 45 years	Yes	No	No	No	No	Underweight	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	Yes	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	Yes	Normal	No	DS
Male	< 45 years	Yes	Yes	Yes	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	Yes	Normal	No	DS
Male	>= 45 years	Yes	No	No	No	Yes	Underweight	No	DS
Female	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	Yes	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	No	No	No	Yes	Normal	No	DS
Female	< 45 years	No	No	No	No	No	Normal	No	DS
Male	>= 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Female	< 45 years	Yes	Yes	No	No	No	Underweight	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	< 45 years	Yes	No	No	No	No	Underweight	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	No	No	No	No	Normal	No	DS
Female	< 45 years	Yes	No	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	Yes	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	< 45 years	No	No	No	No	No	Underweight	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS
Male	>= 45 years	No	Yes	No	No	No	Normal	No	DS
Male	< 45 years	No	Yes	No	No	No	Normal	No	DS
Female	< 45 years	No	No	No	No	No	Normal	No	DS
Female	>= 45 years	No	No	No	No	No	Normal	No	DS"""

# Load the data into a DataFrame
df = pd.read_csv(StringIO(data_str), sep='\t')

# Define mappings to convert categorical to binary
mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Age': {'>= 45 years': 1, '< 45 years': 0},
    'Contact DR': {'Yes': 1, 'No': 0},
    'Smoking': {'Yes': 1, 'No': 0},
    'Alcohol': {'Yes': 1, 'No': 0},
    'Cavitary pulmonary': {'Yes': 1, 'No': 0},
    'Diabetes': {'Yes': 1, 'No': 0},
    'Nutritional': {'Underweight': 1, 'Normal': 0},
    'TB outside': {'Yes': 1, 'No': 0},
    'Class': {'DR': 1, 'DS': 0}
}

# Note: The header has 'TB outside' but in code it's used as is.

# Create reverse mappings
reverse_mappings = {col: {v: k for k, v in mappings[col].items()} for col in mappings}

# Convert dataframe to binary
binary_df = df.copy()
for col in binary_df.columns:
    binary_df[col] = binary_df[col].map(mappings[col])

# Prepare data for torch
input_dim = binary_df.shape[1]
latent_dim = 5  # Choose a latent dimension, small for small data
data_tensor = torch.tensor(binary_df.values, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Initialize model and optimizer
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the model
epochs = 500  # Increase epochs for better learning on small data
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in loader:
        x = batch[0]
        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(loader.dataset):.4f}')

# Generate synthetic data
def generate_samples(model, num_samples, latent_dim):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        samples = model.decoder(z)
        # Use Bernoulli sampling for more variability
        samples = torch.bernoulli(samples)
    return samples

syn_samples = generate_samples(model, 5000, latent_dim)

# Convert back to original categorical values
syn_df = pd.DataFrame(syn_samples.numpy(), columns=df.columns)
for col in syn_df.columns:
    syn_df[col] = syn_df[col].apply(lambda x: reverse_mappings[col][int(x)])

# Save to CSV
syn_df.to_csv('synthetic_tb_data.csv', index=False)
print("Synthetic dataset saved to 'synthetic_tb_data.csv' with 5000 samples.")