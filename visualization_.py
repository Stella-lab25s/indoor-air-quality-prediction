from datetime import timedelta

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from main import AirQualityAnalysis


class AirQualityVisualizer:
    def __init__(self, data=None, data_path='Bedroom.csv'):
        """Initialize the visualization class"""
        self.analyzer = AirQualityAnalysis(data_path)
        self.data = self.analyzer.load_and_preprocess()

        # Store analysis result
        self.vent_results = self.analyzer.analyze_ventilation_effectiveness()

        # Output directory
        self.output_dir = "output/figures"

        # Setup plotting style
        self.set_plot_style()

    def set_plot_style(self):
        """ Set matplotlib style """
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 2,
            'figure.dpi': 100
        })

    def plot_daily_co2_pattern(self):
        """Plot daily CO2 pattern"""
        fig, ax = plt.subplots()

        hourly_stats = self.data.groupby('hour')['co2'].agg(['mean', 'std'])

        ax.plot(hourly_stats.index, hourly_stats['mean'], 'b-', label='Average CO2')
        ax.fill_between(hourly_stats.index,
                        hourly_stats['mean'] - hourly_stats['std'],
                        hourly_stats['mean'] + hourly_stats['std'],
                        alpha=0.2,
                        color='b')

        ax.set_title('Daily CO2 Pattern')
        ax.set_xlabel('Hour')
        ax.set_ylabel('CO2 (ppm)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.savefig(f"{self.output_dir}/daily_co2_pattern.png")
        print(f"    Average daily CO2 concentration: {hourly_stats['mean'].mean():.1f} ppm")
        print(f"    Peak CO2 Period: {hourly_stats['mean'].idxmax()}:00")
        print(f"    Lowest CO2 Period: {hourly_stats['mean'].idxmin()}:00")
        plt.close()

    def plot_ventilation_analysis(self):
        """Draw ventilation analysis graph"""
        events = self.vent_results['events']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # First subplot: Ventilation duration vs CO2 reduction
        durations = [event['duration'].total_seconds() / 60 for event in events]
        reductions = [event['co2_reduction'] for event in events]

        ax1.scatter(durations, reductions, alpha=0.6)
        ax1.set_title('Ventilation Duration vs CO2 Reduction')
        ax1.set_xlabel('Duration (minutes)')
        ax1.set_ylabel('CO2 Reduction (ppm)')
        ax1.grid(True, alpha=0.3)

        # Second subplot: Distribution of ventilation events
        event_hours = [event['start_time'].hour for event in events]
        ax2.hist(event_hours, bins=24, alpha=0.7, color='b')
        ax2.set_title('Distribution of Ventilation Events')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Number of Events')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/ventilation_analysis.png")
        print(f"    Total Ventilation Events: {len(events)}")
        print(f"    Average Duration: {np.mean(durations):.1f} minutes")
        print(f"    Average CO2 Reduction: {np.mean(reductions):.1f} ppm")
        plt.close()

    def plot_correlation_matrix(self):
        """Draw correlation matrix"""
        env_params = ['co2', 'temp', 'humid', 'voc', 'pm25', 'pm10']
        corr_matrix = self.data[env_params].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

        # Add correlation values
        for i in range(len(env_params)):
            for j in range(len(env_params)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")

        plt.colorbar(im)
        ax.set_xticks(range(len(env_params)))
        ax.set_yticks(range(len(env_params)))
        ax.set_xticklabels(env_params)
        ax.set_yticklabels(env_params)
        plt.title('Environmental Parameters Correlation')

        plt.savefig(f"{self.output_dir}/parameter_correlations.png")
        print("    Correlation matrix generated")
        plt.close()

    def plot_environmental_correlations(self):
        """Plot environmental correlation"""
        env_params = ['co2', 'temp', 'humid', 'voc', 'pm25', 'pm10']
        corr_matrix = self.data[env_params].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Environmental Parameters Correlation')

        plt.savefig(f"{self.output_dir}/parameter_correlations.png")
        plt.close()

    def plot_time_series(self, days=7):
        """Draw time series plots"""
        end_time = self.data.index.max()
        start_time = end_time - timedelta(days=days)
        recent_data = self.data[start_time:end_time]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        ax1.plot(recent_data.index, recent_data['co2'], 'b-')
        ax1.set_title(f'CO2 Levels - Last {days} Days')
        ax1.set_ylabel('CO2 (ppm)')
        ax1.grid(True, alpha=0.3)

        ax2.plot(recent_data.index, recent_data['temp'], 'r-', label='Temperature')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(recent_data.index, recent_data['humid'], 'b-', label='Humidity')

        ax2.set_ylabel('Temperature (°C)', color='r')
        ax2_twin.set_ylabel('Humidity (%)', color='b')
        ax2.grid(True, alpha=0.3)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/time_series.png")
        print(f"    Analyzed data for the last {days} days")
        plt.close()

    def create_all_plots(self):
        """Generate all plots"""
        print("\nStarting visualization generation...")

        print("\n1. Generating CO2 daily pattern plot...")
        self.plot_daily_co2_pattern()

        print("\n2. Generating ventilation analysis plots...")
        self.plot_ventilation_analysis()

        print("\n3. Generating parameter correlation matrix...")
        self.plot_correlation_matrix()

        print("\n4. Generating time series plots...")
        self.plot_time_series()

        print(f"\nAll plots saved in: {self.output_dir}")


if __name__ == "__main__":
    # Run sample
    visualizer = AirQualityVisualizer()
    visualizer.create_all_plots()