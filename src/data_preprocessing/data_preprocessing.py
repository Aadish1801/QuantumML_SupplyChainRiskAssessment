import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_supply_chain_data(input_path='data/dynamic_supply_chain_logistics_dataset.csv', features_output_path='data/processed_features.csv', target_output_path='data/processed_target.csv'):
    """
    Loads, preprocesses, and saves the supply chain dataset.
    """
    # Load the dataset
    df = pd.read_csv(input_path)

    # Convert timestamp to datetime, but drop it for the modeling phase as it's an object type
    # We can incorporate time-based features later if needed.
    df = df.drop('timestamp', axis=1)

    # Remove features that would cause data leakage
    leaky_features = ['disruption_likelihood_score', 'delay_probability', 'delivery_time_deviation']
    df = df.drop(columns=leaky_features)
    print(f"Removed leaky features: {leaky_features}")

    # Separate target variable
    target = df['risk_classification']
    features = df.drop('risk_classification', axis=1)

    # Encode the categorical target variable
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)
    
    print("Risk classification mapping:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"- {class_name}: {i}")

    # Identify numerical columns
    numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns

    # Scale numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[numerical_cols])

    # Create a new DataFrame with scaled features
    scaled_features_df = pd.DataFrame(scaled_features, columns=numerical_cols, index=features.index)
    
    # Save the processed data
    scaled_features_df.to_csv(features_output_path, index=False)
    pd.DataFrame(encoded_target, columns=['risk_classification']).to_csv(target_output_path, index=False)

    print(f"\nProcessed features saved to {features_output_path}")
    print(f"Processed target saved to {target_output_path}")

if __name__ == '__main__':
    preprocess_supply_chain_data()
