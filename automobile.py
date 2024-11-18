import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import pickle
import joblib
from sklearn.metrics import mean_squared_error, r2_score

class AutoMobile:
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0)
        }
        self.numeric_features = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 
                               'engine-size', 'bore', 'stroke', 'compression-ratio', 
                               'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg']
        self.categorical_features = ['fuel-type', 'aspiration', 'num-of-doors', 
                                   'body-style', 'drive-wheels', 'engine-location', 
                                   'engine-type', 'num-of-cylinders', 'fuel-system']
        self.cat_label = ["num-of-doors", "num-of-cylinders"]
        self.cat_one = ['fuel-type', 'aspiration', 
                                   'body-style', 'drive-wheels', 'engine-location', 
                                   'engine-type', 'fuel-system','make']
        
    def create_preprocessing_pipeline(self, df):
        # Create a copy of the dataframe
        df = df.copy()
        
        # Handle missing values
        df = df.replace('?', np.nan)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['price'] = df['price'].fillna(df['price'].mean()) 
        # Convert numeric columns
        for feature in self.numeric_features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            # Fix the inplace warning by reassignment
            df[feature] = df[feature].fillna(df[feature].mean())
        
        # Handle categorical features
        for feature in self.categorical_features:
            # Fix the inplace warning by reassignment
            df[feature] = df[feature].fillna(df[feature].mode()[0])
            if feature not in self.encoders:
                self.encoders[feature] = LabelEncoder()
            df[feature] = self.encoders[feature].fit_transform(df[feature])
        
        return df

    def feature_engineering(self, df):
        # Create a copy of the dataframe
        df = df.copy()
        
        # Create new features
        df['weight_power_ratio'] = df['curb-weight'] / df['horsepower']
        df['engine_efficiency'] = df['horsepower'] / df['engine-size']
        df['acceleration_proxy'] = df['horsepower'] / df['curb-weight']
        df['avg_mpg'] = (df['city-mpg'] + df['highway-mpg']) / 2
        df['car_volume'] = df['length'] * df['width'] * df['height']
        df['engine_stress'] = df['compression-ratio'] / df['horsepower']
        df['engine_displacement'] = (np.pi / 4) * (df['bore'] ** 2) * df['stroke'] * df['num-of-cylinders']
        
        
        return df

    def create_ensemble_model(self, X, y):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self, X_train, y_train):
        trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        return trained_models

    def ensemble_predictions(self, models, X):
        predictions = []
        for name, model in models.items():
            pred = model.predict(X)
            predictions.append(pred)
        return np.mean(predictions, axis=0)

    def train(self, df):
        print("Starting preprocessing...")
        # Preprocess the data
        df_processed = self.create_preprocessing_pipeline(df)
        
        print("Feature engineering...")
        df_processed = self.feature_engineering(df_processed)
        
        # Prepare features and target
        features = self.numeric_features + self.categorical_features + ['weight_power_ratio', 'engine_efficiency', 'acceleration_proxy']
        X = df_processed[features]
        y = df_processed['price'].astype(float)
        
        print("Creating and training ensemble model...")
        # Create and train ensemble model
        X_train_scaled, X_test_scaled, y_train, y_test = self.create_ensemble_model(X, y)
        trained_models = self.train_models(X_train_scaled, y_train)
        
        # Save the models and preprocessing objects
        print("Saving models and preprocessing objects...")
        joblib.dump(self.scaler, 'processing.pkl')
        joblib.dump((trained_models, self.encoders, features), 'model.pkl')
        
        # Calculate and return test set performance
        test_predictions = self.ensemble_predictions(trained_models, X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        r2 = r2_score(y_test, test_predictions)
        
        return {
            'rmse': rmse,
            'r2': r2,
            'test_predictions': test_predictions,
            'actual_values': y_test
        }

    def predict(self, input_data):
        # Load the models and preprocessing objects
        scaler = joblib.load('processing.pkl')
        trained_models, encoders, features = joblib.load('model.pkl')
        
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocess the input
        for feature in self.categorical_features:
            if feature in df.columns:
                df[feature] = encoders[feature].transform(df[feature])
        
        df = self.feature_engineering(df)
        
        # Scale the features
        X = df[features]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = self.ensemble_predictions(trained_models, X_scaled)
        return prediction[0]

def main():
    """
    Main function to demonstrate the usage of the AutoMobile class
    """
    # Sample usage
    # Load the data
    df = pd.read_csv('automobile-dataset.txt')  # Replace with your data path
    
    # Initialize the model
    model = AutoMobile()
    
    # Train the model and get performance metrics
    print("Training the model...")
    results = model.train(df)
    
    # Print results
    print("\nModel Performance Metrics:")
    print(f"RMSE: ${results['rmse']:,.2f}")
    print(f"R² Score: {results['r2']:.3f}")
    
    # Example prediction
    sample_input = {
        'fuel-type': 'gas',
        'aspiration': 'std',
        'num-of-doors': 'two',
        'body-style': 'convertible',
        'drive-wheels': 'rwd',
        'engine-location': 'front',
        'wheel-base': 88.6,
        'length': 168.8,
        'width': 64.1,
        'height': 48.8,
        'curb-weight': 2548,
        'engine-type': 'dohc',
        'num-of-cylinders': 'four',
        'engine-size': 130,
        'fuel-system': 'mpfi',
        'bore': 3.47,
        'stroke': 2.68,
        'compression-ratio': 9.0,
        'horsepower': 111,
        'peak-rpm': 5000,
        'city-mpg': 21,
        'highway-mpg': 27
    }
    
    print("\nMaking a sample prediction...")
    predicted_price = model.predict(sample_input)
    print(f"Predicted Price: ${predicted_price:,.2f}")
    
    return model

# For Jupyter notebook usage
if __name__ == "__main__":
    model = main()