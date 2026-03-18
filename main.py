import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ml_model import MachineLearningModels


# Main class
class AirQualityAnalysis:
    def __init__(self, data_path: str = "Bedroom.csv"):
        """ Initialization main analysis class """
        self.data = None
        self.data_path = data_path
        self.statistical_analyzer = None
        self.rf_model = None

    def load_and_preprocess(self):
        """ Load and preprocess data """
        try:
            # 1. Load basic data
            self.data = pd.read_csv(self.data_path)
            print("Column name of CSV file: ", self.data.columns.tolist())

            # 2. Time convert
            timestamp_col = 'timestamp(Asia/Shanghai)'
            # Convert into UK time, setup it as direct index
            self.data.index = (pd.to_datetime(self.data[timestamp_col])
                                    .dt.tz_localize('Asia/Shanghai')
                                    .dt.tz_convert('Europe/London')
                                    .dt.tz_localize(None))

            # 3. Update index name as UK time
            self.data.index.name = 'timestamp(London)'

            # Delete original timestamp column
            if timestamp_col in self.data.columns:
                self.data = self.data.drop(timestamp_col, axis=1)

            # 4. Feature project: add time feature
            self.data['hour'] = self.data.index.hour
            self.data['day_of_week'] = self.data.index.dayofweek

            # 5. Missing value processing
            self.data = self.data.interpolate(method='ffill')

            print("\nSample after preprocessing:")
            print(self.data.head())
            print("\nData Columns:", self.data.columns.tolist())

            return self.data

        except Exception as e:
            print(f"Error loading data : {str(e)}")
            raise


    def analyze_daily_pattern(self, col='co2'):
        """ Analysis daily pattern, Column: CO2 """
        try:
            if self.data is None:
                self.load_and_preprocess()

                print(f"\nAnalysing {col} 's daily pattern...")
                print(f"Acceptable column: {self.data.column.tolist()}")

            # Statistical of hourly
            hourly_stats = self.data.groupby('hour')[col].agg([
                    'mean', 'std', 'min', 'max',
                    'count'  # add datapoint count
            ])

            # Indentify peak hour and low hours period
            peak_hours = hourly_stats['mean'].nlargest(3).index.tolist()
            low_hours = hourly_stats['mean'].nsmallest(3).index.tolist()

            # Calculate variation range daily
            daily_range = {
                'average_range': hourly_stats['max'] - hourly_stats['min'],
                'peak_value': hourly_stats['max'].max(),
                'lowest_value': hourly_stats['min'].min(),
            }

            return {
                'hourly_stats': hourly_stats,
                'peak_hours': peak_hours,
                'low_hours': low_hours,
                'daily_range': daily_range
            }

        except Exception as e:
            print(f"Error in daily pattern analysis: {str(e)}")
            raise


    def analyze_short_term_trend(self, column = 'co2', window = 24):
        """Analysis short term trend(2 hours default)
            window: Rolling window size (5 minutes per point)"""

        # Calculate move average
        rolling_mean = self.data[column].rolling(window=window).mean()

        # Calculate rate of change
        rate_of_change = self.data[column].diff(window)

        # Indentify significant changes feature
        significant_changes = pd.DataFrame({
            'value': self.data[column],
            'rolling_mean': rolling_mean,
            'change_rate': rate_of_change
        })

        # Define significant change threshold
        threshold = rate_of_change.std() * 2

        significant_points = significant_changes[
            (significant_changes['change_rate'].abs() > threshold)
        ]

        return {
            'moving_average': rolling_mean,
            'rate_of_change': rate_of_change,
            'significant_points': significant_points,
            'statistics': {
                'mean_change_rate': rate_of_change.mean(),
                'std_change_rate': rate_of_change.std(),
                'significant_changes_count': len(significant_points)
            }
        }

    def analyze_ventilation_effectiveness(self):
        """ Analyze ventilation effectiveness """
        # Define the parameters of ventilation events
        try:
            if self.data is None:
                self.load_and_preprocess()

            # Calculate CO2 variation
            self.data['co2_change'] = self.data['co2'].diff()

            co2_decrease_threshold = -50  # ppm, negative values express reduction of concentration
            min_duration = pd.Timedelta(minutes=10)  # Shortest ventilated time
            # Indentify ventilation events
            ventilation_events = []
            current_event = None

            for idx, row in self.data.iterrows():
                if row['co2_change'] < co2_decrease_threshold and current_event is None:
                    current_event = {
                        'start_time': idx,
                        'start_co2': row['co2'],
                        'min_co2': row['co2']
                    }
                elif current_event is not None:
                    if row['co2'] < current_event['min_co2']:
                        current_event['min_co2'] = row['co2']

                    if row['co2_change'] >= 0:
                        current_event['end_time'] = idx
                        current_event['end_co2'] = row['co2']
                        duration = current_event['end_time'] - current_event['start_time']

                        if duration >= min_duration:
                            current_event['duration'] = duration
                            current_event['co2_reduction'] = current_event['start_co2'] - current_event['end_co2']
                            current_event['reduction_rate'] = current_event['co2_reduction'] / duration.total_seconds()
                            ventilation_events.append(current_event)

                        current_event = None

            if ventilation_events:
                effectiveness_stats = {
                    'total_events': len(ventilation_events),
                    'avg_duration': np.mean([event['duration'].total_seconds() / 60 for event in ventilation_events]),
                    'avg_reduction': np.mean([event['co2_reduction'] for event in ventilation_events]),
                    'avg_reduction_rate': np.mean([event['reduction_rate'] for event in ventilation_events]),
                    'best_reduction': max([event['co2_reduction'] for event in ventilation_events]),
                    'best_time':
                        ventilation_events[np.argmax([event['co2_reduction'] for event in ventilation_events])][
                            'start_time'].hour
                }
            else:
                effectiveness_stats = {
                    'total_events': 0,
                    'avg_duration': 0,
                    'avg_reduction': 0,
                    'avg_reduction_rate': 0,
                    'best_reduction': 0,
                    'best_time': None,
                    'min_reduction': 0,
                    'median_reduction': 0,
                    'avg_start_co2': 0,
                    'avg_end_co2': 0
                }

            print("\nIdentified ventilation events numbers:", effectiveness_stats['total_events'])
            print("Average ventilate time : {:.1f} mins".format(effectiveness_stats['avg_duration']))
            print("Average reduction amount of CO2: {:.1f} ppm".format(effectiveness_stats['avg_reduction']))

            return {
                'events': ventilation_events,
                'stats': effectiveness_stats
            }

        except Exception as e:
            print(f"Error analysing of ventilation analysis: {str(e)}")
            raise

# Model Evaluation Class
class ModelEvaluationSystem:
    def __init__(self):
        """ Initialization """
        self.metrics = {}
        self.predictions = {}

    def evaluate_model(self, y_true, y_pred, model_name = "default"):
        """ Model Evaluation"""
        # Store predicted result
        self.predictions[model_name] = {
            'true': y_true,
            'pred': y_pred
        }

        # Calculate evaluation indicators
        mse = mean_squared_error(y_true, y_pred) # Mean Squared Error
        rmse = np.sqrt(mse) # Root Mean Squared Error
        mae = mean_absolute_error(y_true, y_pred) # Mean absolute Error
        r2 = r2_score(y_true, y_pred) # R2 score

        # Calculate predicted interval
        residuals = y_true - y_pred
        residuals_std = np.std(residuals)

        # 95% predicted interval
        prediction_interval = 1.96 * residuals_std

        self.metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Prediction_Interval': prediction_interval
        }

        return self.metrics[model_name]

def main():
    # Train machine learning model
    ml_model = MachineLearningModels()
    ml_model.prepare_data()
    ml_predictions = ml_model.train_model()

    # Evaluate in details through 'ModelEvaluationSystem'
    evaluator = ModelEvaluationSystem()
    evaluator.evaluate_model(
        y_true=ml_model.y_test,
        y_pred=ml_predictions,
        model_name="RandomForest"
    )


# Run analysis sample
if __name__ == "__main__":
    # Create an instance and run analysis
    analyzer = AirQualityAnalysis()
    try:
        print("\n Starting data preprocessing...")
        data = analyzer.load_and_preprocess()

        if data is not None:
            print("\n Starting ventilation effectiveness analysis...")
            vent_result = analyzer.analyze_ventilation_effectiveness()
            print("Total events:", vent_result['stats']['total_events'])
            print("Average duration: {:.2f} minutes".format(vent_result['stats']['avg_duration']))
            print("Average CO2 reduction: {:.2f} ppm".format(vent_result['stats']['avg_reduction']))
            print("Average reduction rate: {:.4f} ppm/s".format(vent_result['stats']['avg_reduction_rate']))
            print("Best reduction: {:.2f} ppm".format(vent_result['stats']['best_reduction']))
            print("Best time: {:02d}:00".format(vent_result['stats']['best_time']))

            print("\nStarting analysis CO2 daily pattern...")
            daily_results = analyzer.analyze_daily_pattern()
            print("\nCO2 DAILY STATISTIC:")
            print(daily_results['hourly_stats'])
            print("\nPeak period:", daily_results['peak_hours'])
            print("Low period:", daily_results['low_hours'])

    except Exception as e:
        print(f"Error occur during processing: {str(e)} ")