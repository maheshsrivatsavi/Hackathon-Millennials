import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Synthetic data generation function
def generate_synthetic_data(num_records, base_date, anomaly_rate=0.2):
    transaction_ids = range(1, num_records + 1)
    dates = [base_date + timedelta(days=i) for i in range(num_records)]
    amounts = np.random.normal(loc=500, scale=200, size=num_records).clip(min=50, max=1000)
    num_anomalies = int(num_records * anomaly_rate)
    anomaly_indices = np.random.choice(num_records, num_anomalies, replace=False)
    amounts[anomaly_indices] = np.random.uniform(5000, 10000, num_anomalies)  # Strong anomalies
    data = pd.DataFrame({
        'Transaction_ID': transaction_ids,
        'Date': dates,
        'Amount': amounts
    })
    return data

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

def train_vae(data, epochs=50):
    # Normalize data to 0-1 range
    min_val = float(data['Amount'].min())  # Convert to scalar
    max_val = float(data['Amount'].max())  # Convert to scalar
    normalized_data = (data - min_val) / (max_val - min_val)
    
    input_dim = normalized_data.shape[1]
    vae = VAE(input_dim=input_dim, hidden_dim=16, latent_dim=8)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    data_tensor = torch.FloatTensor(normalized_data.values)
    
    for epoch in range(epochs):
        vae.zero_grad()
        recon_data, mu, logvar = vae(data_tensor)
        loss = vae_loss(recon_data, data_tensor, mu, logvar)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        recon_data, _, _ = vae(data_tensor)
        recon_error = torch.mean((recon_data - data_tensor) ** 2, dim=1)
        threshold = recon_error.mean() + 1 * recon_error.std()
        anomalies = recon_error > threshold
    
    # Denormalize reconstruction errors for display
    recon_error = recon_error * (max_val - min_val)  # Now both are scalars
    return anomalies.numpy(), recon_error.numpy(), min_val, max_val

# Streamlit UI
def main():
    st.title("Anomaly Detection with Variational Autoencoder")

    # Sidebar for parameters
    st.sidebar.header("Settings")
    num_records = st.sidebar.slider("Number of Transactions", 10, 100, 50)
    anomaly_rate = st.sidebar.slider("Anomaly Rate", 0.0, 0.5, 0.2, step=0.05)
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, step=10)

    # Generate data
    base_date = datetime(2025, 3, 1)
    data1 = generate_synthetic_data(num_records, base_date, anomaly_rate)
    data2 = data1.copy()
    discrepancy_indices = np.random.choice(num_records, int(num_records * 0.2), replace=False)
    data2.loc[discrepancy_indices, 'Amount'] = data2.loc[discrepancy_indices, 'Amount'] * np.random.uniform(0.8, 1.2, len(discrepancy_indices))

    data1['Amount'] = data1['Amount'].round(2)
    data2['Amount'] = data2['Amount'].round(2)

    # Combine data for VAE
    combined_data = pd.concat([data1[['Amount']], data2[['Amount']]], axis=0)

    # Run VAE when button is clicked
    if st.button("Detect Anomalies"):
        with st.spinner("Training VAE and detecting anomalies..."):
            anomalies, recon_errors, min_val, max_val = train_vae(combined_data, epochs=epochs)
            combined_data['Reconstruction_Error'] = recon_errors
            combined_data['Is_Anomaly'] = anomalies

            # Debug output
            st.write("Debug: Number of anomalies detected:", anomalies.sum())
            st.write("Debug: Threshold:", (recon_errors.mean() + 1 * recon_errors.std()))

            # Display results
            st.subheader("All Transactions")
            st.write(combined_data)

            st.subheader("Detected Anomalies")
            anomaly_data = combined_data[combined_data['Is_Anomaly']]
            if not anomaly_data.empty:
                st.write(anomaly_data)
            else:
                st.write("No anomalies detected. Try increasing anomaly rate or lowering threshold.")

            # Visualization
            st.subheader("Reconstruction Error Plot")
            fig, ax = plt.subplots()
            ax.scatter(range(len(recon_errors)), recon_errors, c=['red' if a else 'blue' for a in anomalies], alpha=0.5)
            ax.axhline(y=recon_errors.mean() + 1 * recon_errors.std(), color='green', linestyle='--', label='Threshold')
            ax.set_xlabel("Transaction Index")
            ax.set_ylabel("Reconstruction Error")
            ax.legend()
            st.pyplot(fig)

    # Option to download data
    st.sidebar.subheader("Download Data")
    if st.sidebar.button("Save Datasets"):
        data1.to_csv('synthetic_bank_statement1.csv', index=False)
        data2.to_csv('synthetic_bank_statement2.csv', index=False)
        st.sidebar.success("Datasets saved as 'synthetic_bank_statement1.csv' and 'synthetic_bank_statement2.csv'")

if __name__ == "__main__":
    main()