import pandas as pd
from datetime import datetime
import pytz


def convert_timezone(csv_file):
    # Read the CSV file
    df = pd.read_csv('Bedroom.csv')

    # Parse the timestamp column
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    london_tz = pytz.timezone('Europe/London')

    # Convert timestamps
    def convert_time(timestamp_str):
        # Parse the timestamp with Shanghai timezone
        dt = pd.to_datetime(timestamp_str).tz_localize(shanghai_tz)
        # Convert to London time
        london_time = dt.tz_convert(london_tz)
        # Format the output
        return london_time.strftime('%Y-%m-%d %H:%M:%S')

    # Apply conversion to timestamp column
    df['timestamp(BST)'] = df['timestamp(Asia/Shanghai)'].apply(convert_time)

    # Reorder columns to put new timestamp first
    cols = df.columns.tolist()
    cols = ['timestamp(BST)'] + [col for col in cols if col != 'timestamp(BST)']
    df = df[cols]

    # Save to new CSV
    output_file = 'Bedroom_BST.csv'
    df.to_csv(output_file, index=False)
    return output_file


# Example usage
output = convert_timezone('Bedroom.csv')
print(f"Converted file saved as: {output}")