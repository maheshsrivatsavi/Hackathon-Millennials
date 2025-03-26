import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from unittest.mock import patch
import io
import sys

# Define functions
def generate_synthetic_data(num_records, base_date, anomaly_rate=0.2):
    transaction_ids = range(1, num_records + 1)
    dates = [base_date + timedelta(days=i) for i in range(num_records)]
    amounts = np.random.normal(loc=500, scale=200, size=num_records).clip(min=50, max=1000)
    num_anomalies = int(num_records * anomaly_rate)
    anomaly_indices = np.random.choice(num_records, num_anomalies, replace=False)
    amounts[anomaly_indices] = np.random.uniform(5000, 10000, num_anomalies)
    data = pd.DataFrame({
        'Transaction_ID': transaction_ids,
        'Date': dates,
        'Amount': amounts
    })
    return data

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

def train_vae(data, epochs=50, threshold_multiplier=1.0):
    min_val = float(data['Amount'].min())
    max_val = float(data['Amount'].max())
    normalized_data = (data - min_val) / (max_val - min_val)
    
    input_dim = normalized_data.shape[1]
    vae = VAE(input_dim=input_dim, hidden_dim=16, latent_dim=8)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
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
        threshold = recon_error.mean() + threshold_multiplier * recon_error.std()
        anomalies = recon_error > threshold
    
    recon_error = recon_error * (max_val - min_val)
    return anomalies.numpy(), recon_error.numpy(), min_val, max_val

# Custom TestResult class to change the output
class CustomTestResult(unittest.TextTestResult):
    def printErrors(self):
        # Override to avoid printing errors if there are none
        if self.errors or self.failures:
            super().printErrors()

    def addSuccess(self, test):
        super().addSuccess(test)

    def wasSuccessful(self):
        return len(self.failures) == 0 and len(self.errors) == 0

    def print_result(self):
        if self.wasSuccessful():
            print(f"{self.testsRun} passed")
        else:
            print(f"FAILED (failures={len(self.failures)}, errors={len(self.errors)})")

# Custom TestRunner to use CustomTestResult
class CustomTestRunner(unittest.TextTestRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(resultclass=CustomTestResult, *args, **kwargs)

    def run(self, test):
        result = super().run(test)
        result.print_result()
        return result

# Test cases
class TestAnomalyDetection(unittest.TestCase):

    def setUp(self):
        self.base_date = datetime(2025, 3, 1)
        self.num_records = 50
        np.random.seed(42)
        torch.manual_seed(42)

    def test_generate_synthetic_data(self):
        data = generate_synthetic_data(self.num_records, self.base_date, anomaly_rate=0.2)
        self.assertEqual(data.shape, (self.num_records, 3))
        self.assertListEqual(list(data.columns), ['Transaction_ID', 'Date', 'Amount'])
        self.assertTrue(all(data['Transaction_ID'] == range(1, self.num_records + 1)))
        expected_dates = [self.base_date + timedelta(days=i) for i in range(self.num_records)]
        self.assertTrue(all(data['Date'] == expected_dates))
        anomalies = data['Amount'] >= 5000
        self.assertTrue(anomalies.sum() > 0)
        self.assertAlmostEqual(anomalies.sum() / self.num_records, 0.2, delta=0.15)

    def test_vae_model(self):
        input_dim, hidden_dim, latent_dim = 1, 16, 8
        vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        x = torch.FloatTensor([[0.5]])
        recon_x, mu, logvar = vae(x)
        self.assertEqual(recon_x.shape, torch.Size([1, input_dim]))
        self.assertEqual(mu.shape, torch.Size([1, latent_dim]))
        self.assertEqual(logvar.shape, torch.Size([1, latent_dim]))

    def test_train_vae(self):
        data = generate_synthetic_data(self.num_records, self.base_date, anomaly_rate=0.2)
        anomalies, recon_errors, min_val, max_val = train_vae(data[['Amount']], epochs=10)
        self.assertIsInstance(anomalies, np.ndarray)
        self.assertIsInstance(recon_errors, np.ndarray)
        self.assertEqual(len(anomalies), self.num_records)
        self.assertEqual(len(recon_errors), self.num_records)
        self.assertTrue(anomalies.sum() >= 0)
        self.assertAlmostEqual(min_val, float(data['Amount'].min()), places=6)
        self.assertAlmostEqual(max_val, float(data['Amount'].max()), places=6)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_no_anomalies_output(self, mock_stdout):
        data = pd.DataFrame({'Amount': np.random.normal(loc=500, scale=10, size=10)})
        anomalies, recon_errors, _, _ = train_vae(data, epochs=5, threshold_multiplier=3.0)
        print(f"Debug: Anomalies detected: {anomalies.sum()}")
        print(f"Debug: Reconstruction errors: {recon_errors}")
        print(f"Debug: Threshold: {recon_errors.mean() + 3.0 * recon_errors.std()}")
        if not anomalies.any():
            print("No anomalies detected. Try increasing anomaly rate or lowering threshold.")
        output = mock_stdout.getvalue()
        self.assertIn("No anomalies detected", output)

if __name__ == '__main__':
    # Use the custom runner instead of unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnomalyDetection)
    CustomTestRunner(verbosity=2).run(suite)