import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class MachineLearningModels:
    def __init__(self, data_path="Bedroom.csv"):
        """ Initialization model class """
        self.data = None
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.predictions = None

    def load_and_preprocess(self):
        """ Load and process data """
        try:
            # 1. Load data
            self.data = pd.read_csv(self.data_path)

            # 2. Time processing
            timestamp_col = 'timestamp(Asia/Shanghai)'
            self.data.index = (pd.to_datetime(self.data[timestamp_col])
                               .dt.tz_localize('Asia/Shanghai')
                               .dt.tz_convert('Europe/London')
                               .dt.tz_localize(None))

            # Delete original series
            if timestamp_col in self.data.columns:
                self.data = self.data.drop(timestamp_col, axis=1)

            # 3. Add basic time feature
            self.data['hour'] = self.data.index.hour
            self.data['day_of_week'] = self.data.index.dayofweek

            # 4. Covert numeric type
            numeric_columns = ['co2', 'temp', 'humid', 'voc', 'pm25', 'pm10', 'score']
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            # 5. Feature engineering
            self._feature_engineering()
            print("Data preprocessing completed successfully")

            return self.data

        except Exception as e:
            print(f"Loading data error: {str(e)}")
            raise

    def _feature_engineering(self):
        """ Feature Engineering """
        try:
            # Time cycle feature
            self.data['hour_sin'] = np.sin(self.data['hour'] * (2 * np.pi / 24))
            self.data['hour_cos'] = np.cos(self.data['hour'] * (2 * np.pi / 24))

            # Lagging feature
            for lag in [1, 2, 3]:
                self.data[f'co2_lag_{lag}'] = self.data['co2'].shift(lag)

            # Rolling window feature
            self.data['co2_rolling_mean'] = self.data['co2'].rolling(window=6).mean()
            self.data['co2_rolling_std'] = self.data['co2'].rolling(window=6).std()

            # Environmental reactive feature
            self.data['temp_humid_interaction'] = self.data['temp'] * self.data['humid']

            # Missing value processing
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')

            # Check missing value
            if self.data.isnull().values.any().any():
                print('Warning: Still have missing values after imputation.')
                # fill if it existed
                self.data = self.data.fillna(self.data.mean())

            print("Feature engineering completed successfully")

        except Exception as e:
            print(f"Feature engineering error: {str(e)}")
            raise

    def prepare_model_data(self, target='co2', test_size=0.2):
        """ Prepare training data for model """
        try:
            # Choose feature
            features = [
                'temp', 'humid', 'voc', 'pm25', 'pm10',
                'hour', 'day_of_week',
                'hour_sin', 'hour_cos',
                'co2_lag_1', 'co2_lag_2', 'co2_lag_3',
                'co2_rolling_mean', 'co2_rolling_std',
                'temp_humid_interaction'
            ]

            # Check existed of all feature
            missing_features = [f for f in features if f not in self.data.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            X = self.data[features]
            y = self.data[target]

            # Check missing value
            if X.isnull().any().any():
                print("Warning: Features contain missing values")
                print("Missing value counts:")
                print(X.isnull().sum())
                # Fill via average value
                X = X.fillna(X.mean())

            # Standard the number
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split training set and test set
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )

            return self.X_train, self.X_test, self.y_train, self.y_test

            # Final check
            assert not np.isnan(self.X_train).any(), "Training data contains NaN"
            assert not np.isnan(self.X_test).any(), "Test data contains NaN"
            assert not np.isnan(self.y_train).any(), "Training labels contain NaN"
            assert not np.isnan(self.y_test).any(), "Test labels contain NaN"

            print("Data preparation completed successfully")
            return self.X_train, self.X_test, self.y_train, self.y_test


        except Exception as e:
            print(f"Preparing data error: {str(e)}")
            raise

    def check_data_quality(self):
        """Check quality of data"""
        print("\nData Quality Report:")
        print("-" * 50)

        # Missing value checking
        missing = self.data.isnull().sum()
        print("\nMissing Values:")
        print(missing[missing > 0])

        # Check the scope of value
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        print("\nValue Ranges:")
        print(self.data[numeric_columns].describe())

        # Outlier check
        print("\nValue Ranges:")
        print(self.data.describe())

        # Check correlations of feature
        print("\nFeature Correlations with CO2:")
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlations = numeric_data.corr()['co2'].sort_values(ascending=False)
        print(correlations)

    def train_random_forest_model(self, n_estimators=100):
        """ RandomForest Model training"""
        try:
            # Check preparation
            if self.X_train is None:
                self.prepare_model_data()

            # Initialization and training model
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1  # Use whole core of CPU
            )

            self.model.fit(self.X_train, self.y_train)

            # Predict
            self.predictions = self.model.predict(self.X_test)

            return self.evaluate_model()

        except Exception as e:
            print(f"Model training error: {str(e)}")
            raise

    def evaluate_model(self):
        """ Model Evaluation """
        try:
            # Compute the evaluation indicators
            mse = mean_squared_error(self.y_test, self.predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, self.predictions)

            # Feature importance
            feature_names = [
                'temp', 'humid', 'voc', 'pm25', 'pm10',
                'hour', 'day_of_week',
                'hour_sin', 'hour_cos',
                'co2_lag_1', 'co2_lag_2', 'co2_lag_3',
                'co2_rolling_mean', 'co2_rolling_std',
                'temp_humid_interaction'
            ]
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            return {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'feature_importance': feature_importance
            }

        except Exception as e:
            print(f"Model evaluating error: {str(e)}")
            raise

    def predict_future(self, input_data):
        """ Predict with trained model """
        try:
            if self.model is None:
                raise ValueError("Haven't train model")

            # Ensure the format of data is correct
            scaler = StandardScaler()
            scaler.fit(self.data[input_data.columns])
            input_scaled = scaler.transform(input_data)

            return self.model.predict(input_scaled)

        except Exception as e:
            print(f"Predicting error: {str(e)}")
            raise

    def main_analysis(self):
        """Main analysis process"""
        print("Starting machine learning analysis...")
        # Load Data
        self.load_and_preprocess()
        # Check data's quality
        self.check_data_quality()
        # Data preparation
        self.prepare_model_data()
        # Model training
        model_results = self.train_random_forest_model()

        print("\nModel Evaluation Results:")
        print(f"MSE: {model_results['mse']:.2f}")
        print(f"RMSE: {model_results['rmse']:.2f}")
        print(f"R² Score: {model_results['r2']:.2f}")
        print("\nFeature Importance:")
        print(model_results['feature_importance'])

        return model_results


# Run sample
if __name__ == "__main__":
    ml_analyzer = MachineLearningModels()
    results = ml_analyzer.main_analysis()