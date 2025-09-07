import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv('../data/sample_transactions.csv')

# Feature selection
features = df[['amount']]

# Train anomaly detection model
model = IsolationForest(contamination=0.2, random_state=42)
df['anomaly_score'] = model.fit_predict(features)

# Mark anomalies
df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

# Save results
df.to_csv('../reports/fraud_detection_results.csv', index=False)
print("Fraud detection completed. Results saved to reports/fraud_detection_results.csv")
