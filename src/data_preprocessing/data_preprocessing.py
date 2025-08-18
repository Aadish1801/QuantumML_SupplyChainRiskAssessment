import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_supply_chain_data(input_path='data/dynamic_supply_chain_logistics_dataset.csv', features_output_path='data/processed_features.csv', target_output_path='data/processed_target.csv'):
    """
    Loads, preprocesses, and saves the supply chain dataset.
    """
    # Load the dataset
    df = pd.read_csv(input_path)

    # Convert timestamp to datetime and extract time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df = df.drop('timestamp', axis=1)

    # Create interaction features and ratios
    df['demand_x_reliability'] = df['historical_demand'] * df['supplier_reliability_score']
    df['congestion_ratio'] = df['traffic_congestion_level'] / (df['port_congestion_level'] + 1) # Add 1 to avoid division by zero
    df['cost_per_lead_day'] = df['shipping_costs'] / (df['lead_time_days'] + 1)

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

    # Explicitly define numerical columns for scaling
    numerical_cols = [
        'vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate',
        'eta_variation_hours', 'traffic_congestion_level', 'warehouse_inventory_level',
        'loading_unloading_time', 'handling_equipment_availability', 'order_fulfillment_status',
        'weather_condition_severity', 'port_congestion_level', 'shipping_costs',
        'supplier_reliability_score', 'lead_time_days', 'historical_demand',
        'iot_temperature', 'cargo_condition_status', 'route_risk_level',
        'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score',
        'hour', 'day_of_week', 'month', 'demand_x_reliability', 'congestion_ratio',
        'cost_per_lead_day'
    ]

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