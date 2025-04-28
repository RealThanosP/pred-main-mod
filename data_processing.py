import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
import os
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

# Load data
#* DONE
def load_data(sensors:list[str], folder_path:os.PathLike, profile_columns:list[str]=[], sep:str="\t", has_profile_file:bool=False, file_extension:str=".txt", profile_filename:str="profile") -> dict | tuple:
    """
    Loads all the sensor data and a profile (if exists) and returns a dictionary.
    If there is a profile file in the dataset (and has_profile_file=True) then it returns a tuple (sensor_dict, profile_df)
    The dictionary has as keys the name of the sensors, and dataframes with the data as values.

    Args:
        sensors (list[str]): list of sensors names (not the full filename with extension)
        folder_path (os.PathLike): Pathlike object or str path
        profile_columns (list[str]): list of the profile_columns, make sure it has the same SIZE as the csv of the profile. Defaults to [].
        sep (str, optional): Separator used in all the files of the dataset. Defaults to "/t".
        has_profile_file (bool, optional): True if dataset has profile file. Defaults to True.
        file_extension (str, optional): File extension of the files of the dataset. Defaults to ".txt".
        profile_filename (str, optional): filename (not with extension) of the profile file of the dataset. Defaults to "profile".

    Returns:
        dict|tuple: Returns a dictionary with sensor names as keys and dataframes as values. If has_profile_file=True then it returns a tuple with the sensor dictionary (index 0) and profile dataframe (index 1)

    Example:
        sensor_dfs_dict, profile_dict = load_data(sensors=sensors, folder_path=folder_path, profile_columns=profile_columns, has_profile_file=True)
        sensor_dfs_dict= load_data(sensors=sensors, folder_path=folder_path)

    """
    # Gets all the text files from dataset
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    
    # Keeps only the data that have a sensor name in the filename
    sensor_file_path_list = [f"{folder_path}/{f}" for f in txt_files if any(sensor in f for sensor in sensors)]
    sensor_file_path_list.sort()

    # Load all the files as separate DataFrames with the file name as name
    sensor_dfs = {}
    for file_path in sensor_file_path_list:
        # remove the .txt
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # read the file and save it with the right name
        sensor_dfs[file_name] = pd.read_csv(file_path, sep="\t", header=None)
    
    # Load profile
    if has_profile_file:
        profile = pd.read_csv(f"{folder_path}/{profile_filename}{file_extension}", sep=sep, header=None)
        profile.columns = profile_columns
        return sensor_dfs, profile
    
    return sensor_dfs # in case no profile is inside the dataset 

# Helpers
# Creates statistical data for the sensors
def create_stats(sensor_name, df):
    # We get the DataFrame dynamically from the name
    # Calculate statistics
    stats = pd.DataFrame({
        'mean'+ sensor_name: df.mean(axis=1),
        'std'+ sensor_name: df.std(axis=1),
        'min'+ sensor_name: df.min(axis=1),
        'max'+ sensor_name: df.max(axis=1),
        'range'+ sensor_name: df.max(axis=1) - df.min(axis=1),
        'rms'+ sensor_name: (df.pow(2).mean(axis=1)).pow(0.5)
    })
    stat_names = [f'mean{sensor_name}', f'std{sensor_name}', f'min{sensor_name}', f'max{sensor_name}', f'range{sensor_name}', f'rms{sensor_name}']
    return stats

# Combines statistics of the same sensor type by adding the stats
def combine_sensor_stats(df_list: list[pd.DataFrame], prefix: str, stat_names: list[str]) -> pd.DataFrame:
    """
    Adds corresponding statistical features (column-wise and row-wise) from multiple DataFrames.

    Parameters:
    - df_list (list[pd.DataFrame]): List of pandas DataFrames with the same shape, columns, and index.
    - prefix (str): Prefix to apply to each statistical feature (e.g., 'PS').
    - stat_names (list[str]): List of statistical feature names (e.g., ['mean', 'std', 'min', 'max', 'range', 'rms']).

    Returns:
    - pd.DataFrame: A new DataFrame with summed columns for each statistical feature, with prefixed column names.
    """
    # Initialize individual stat containers
    mean_df, std_df, min_df, max_df, range_df, rms_df = [pd.DataFrame() for _ in range(6)]

    for df in df_list:
        # Rename columns according to the prefix
        df.columns = [f"{stat}{prefix}" for stat in stat_names]

        # Extract individual columns
        mean = df[f"mean{prefix}"]
        std = df[f"std{prefix}"]
        min_ = df[f"min{prefix}"]
        max_ = df[f"max{prefix}"]
        range_stat = df[f"range{prefix}"]
        rms = df[f"rms{prefix}"]

        # Accumulate sums
        mean_df = pd.concat([mean_df, mean], axis=1).sum(axis=1)
        std_df = pd.concat([std_df, std], axis=1).sum(axis=1)
        min_df = pd.concat([min_df, min_], axis=1).sum(axis=1)
        max_df = pd.concat([max_df, max_], axis=1).sum(axis=1)
        range_df = pd.concat([range_df, range_stat], axis=1).sum(axis=1)
        rms_df = pd.concat([rms_df, rms], axis=1).sum(axis=1)

    # Combine all the summed stats into a final DataFrame
    summed_df = pd.concat([mean_df, std_df, min_df, max_df, range_df, rms_df], axis=1)
    summed_df.columns = [f"{stat}{prefix}" for stat in stat_names]

    return summed_df

# Computes Coefficient of Variance
def compute_coefficient_variance(row, sensors):
    cv_values = {}
    for name in sensors:
        mean = row[f'mean{name}']
        std = row[f'std{name}']
        cv_values[f'cv{name}'] = std / (mean + 1e-6)  # Avoid division by zero
    return cv_values

# Computes Stability ratio
def compute_stability(row, sensors):
    """
    Computes two stability metrics for each sensor:
      1. Range-based: mean / (range + 1e-6)      -> stabilityRange{name}
      2. Margin-based: (U - (mean + alpha*std)) / (U - L + 1e-6) -> stabilityMargin{name}
    """

    stability_values = {}
    epsilon = 1e-6
    alpha = 1.0  # You can tweak this factor if you like

    for name in sensors:
        # 1) Original Range-Based Stability
        mean_val = row[f'mean{name}']
        range_val = row[f'range{name}']
        stability_values[f'stabilityRange{name}'] = mean_val / (range_val + epsilon)

        # 2) Margin-to-Limits Stability (if columns exist)
        #    If you do not have these columns, comment this section out or add them to your data.
        std_col = f'std{name}'
        upper_col = f'upperLimit{name}'
        lower_col = f'lowerLimit{name}'

        if std_col in row and upper_col in row and lower_col in row:
            std_val = row[std_col]
            upper_limit = row[upper_col]
            lower_limit = row[lower_col]

            margin = upper_limit - (mean_val + alpha * std_val)
            if margin < 0:
                margin = 0
            denom = (upper_limit - lower_limit) + epsilon
            stability_values[f'stabilityMargin{name}'] = margin / denom

        # If not found, we simply skip or could set some default

    return stability_values

# Frequency helper functions
# Does fft_analysis
def fft_analysis(data, sampling_rate_hz):
    """
    Perform FFT analysis on each row of raw sensor data.

    Parameters:
    - data (np.ndarray or pd.DataFrame): shape (2205, N), raw time-domain signals.
    - sampling_rate_hz (int): Sampling rate in Hz.

    Returns:
    - pd.DataFrame: DataFrame of FFT magnitudes with frequency bin labels as columns.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    num_instances, num_samples = data.shape

    # Frequency bins: same for all rows
    freqs = rfftfreq(num_samples, d=1/sampling_rate_hz)

    # Apply FFT row-wise, get magnitude spectrum
    fft_magnitudes = np.abs(rfft(data, axis=1))

    # Convert to DataFrame with frequency labels
    fft_df = pd.DataFrame(fft_magnitudes, columns=[f"{round(f, 2)}Hz" for f in freqs])

    return fft_df

# Computes band_power
def band_power(fft_df, freqs, band_limits, name):
    """
    Calculate band power from FFT magnitudes.

    Args:
        fft_df (pd.DataFrame): The DataFrame containing FFT magnitudes.
        freqs (np.ndarray): Array of frequencies.
        band_limits (list): List of frequency band limits.
        name (str): Name of the sensor

    Returns:
        pd.DataFrame: DataFrame with band power features.
    """
    band_features = {}
    for low, high in band_limits:
        band_name = f"{name}{int(low)}_{int(high)}Hz"
        mask = (freqs >= low) & (freqs < high)
        band_features[band_name] = fft_df.iloc[:, mask].mean(axis=1)

    band_df = pd.DataFrame(band_features)
    band_df = band_df.dropna(axis=1, how='any')

    return band_df

# Helper funcion to create correctly the freqs array
def get_freqs(fft_df, sampling_rate_hz):
    n_bins = fft_df.shape[1]  # number of FFT bins (columns)
    freqs = np.fft.rfftfreq(n=2 * (n_bins - 1), d=1 / sampling_rate_hz)
    return freqs

# Computes top frequency peaks
def top_peaks(fft_df, freqs, name, n=3):
    """
    Extracts top N frequency peaks and returns them in a structured DataFrame.

    Parameters:
    - fft_df (pd.DataFrame): FFT magnitudes per instance (rows) x frequency bins (columns)
    - freqs (np.ndarray): frequency bin centers corresponding to FFT columns
    - n (int): Number of top peaks to extract
    - name (str): Name of the sensor

    Returns:
    - pd.DataFrame: with columns [peak1_freq, peak1_mag, peak2_freq, peak2_mag, ...]
    """
    peak_features = []

    for _, row in fft_df.iterrows():
        peaks, _ = find_peaks(row.values, prominence=1)
        peak_mags = row.values[peaks]
        top_indices = np.argsort(peak_mags)[-n:]

        # Sort by descending magnitude for consistency
        sorted_indices = top_indices[np.argsort(-peak_mags[top_indices])]

        peak_freqs = freqs[peaks[sorted_indices]]
        peak_vals = peak_mags[sorted_indices]

        # Pad if fewer than n peaks
        if len(peak_freqs) < n:
            pad_len = n - len(peak_freqs)
            peak_freqs = np.pad(peak_freqs, (0, pad_len), constant_values=np.nan)
            peak_vals = np.pad(peak_vals, (0, pad_len), constant_values=np.nan)

        feature_row = list(np.column_stack((peak_freqs, peak_vals)).flatten())
        peak_features.append(feature_row)

    columns = [f"{name}_peak{i+1}_freq" if j % 2 == 0 else f"{name}_peak{i+1}_mag"
               for i in range(n) for j in range(2)]

    top_peaks_df = pd.DataFrame(peak_features, columns=columns)
    top_peaks_df = top_peaks_df.dropna(axis=1, how='any')

    return top_peaks_df

# Computes spectral_features
def spectral_features(fft_df, freqs, name):
    """
    Calculate spectral centroid, entropy, and flatness for each row in the FFT dataframe.

    Returns:
        pd.DataFrame with columns: [spectral_centroid, spectral_entropy, spectral_flatness]
    """
    spectral_data = []

    for _, row in fft_df.iterrows():
        magnitudes = row.values
        mag_sum = np.sum(magnitudes)

        # Avoid divide-by-zero for silent signal rows
        if mag_sum == 0:
            centroid = np.nan
            entropy = np.nan
            flatness = np.nan
        else:
            probs = magnitudes / mag_sum

            # Spectral Centroid
            centroid = np.sum(freqs * probs)

            # Spectral Entropy
            entropy = -np.sum(probs * np.log2(probs + 1e-10))  # Add epsilon to avoid log(0)

            # Spectral Flatness
            geometric_mean = np.exp(np.mean(np.log(magnitudes + 1e-10)))
            arithmetic_mean = np.mean(magnitudes)
            flatness = geometric_mean / (arithmetic_mean + 1e-10)

        spectral_data.append([centroid, entropy, flatness])

    return pd.DataFrame(spectral_data, columns=[f"{name}_spectral_centroid", f"{name}_spectral_entropy", f"{name}_spectral_flatness"])

#! NEEDS TESTING and it's probably REALLY slow
# Process data
# Returns a dataframe that the model can work with to predict.
# Also prints the states of the training_df.
# The targets are not included in the returning dataframe
def process_data(sensors:list[str], sensor_dict:dict[str, pd.DataFrame], sensors_duplicates:dict[str, int], stat_names:list[str], sampling_rates_dict:dict[str, int], 
                 sensors_for_statistics:list[str], scaler:StandardScaler | MinMaxScaler, profile_df:pd.DataFrame=[]):
    """
    Processes the data from the sensor_dict and the profile_df and extract statistical, and frequency features that are added to the final 
    reference dataframe X.

    Args:
        sensors (list[str]): List of the sensors used in the setup from the dataset.
        sensor_dict (dict[str, pd.DataFrame]): Sensor dict as explained in the load_data function.
        sensors_duplicates (dict[str, int]): Dictionary with the prefixes of the duplicate sensors in the dataset as keys and values the number of the sensor repeating in the dataset.
        stat_names (list[str]): A list with the statistics that will be added. Used for column adding.
        sampling_rates_dict (dict[str, int]): Dictionary with all the sensor names (str) as keys and their respecting sampling rate as values.
        sensors_for_statistics (list[str]): Short list of names of sensors that end up in the statistics dataframe.
        scaler (StandardScaler | MinMaxScaler): Scaler from sklearn.preprocessing module.
        profile_df (pd.DataFrame): Profile dataframe as returned from the load_data(). Defaults to [].

    Returns:
        pd.DataFrame: Returns a dataframe with the reference columns of the model, without the target columns on it. 
    """
    # Statistical analysis
    # Create statistical dict(str:df)
    stats_df = {}
    for sensor in sensors:
        stats_df[f'{sensor}_stats'] = create_stats(sensor, sensor_dict[sensor])

    # Delete the "duplicate" sensors and replace with combined data
    
    for sensor in sensors_duplicates:
        temp = []
        for i in range(1, sensors_duplicates[sensor] + 1):
            temp.append(stats_df[f"{sensor}{i}_stats"])
            stats_df.pop(f"{sensor}{i}_stats") # delete the individual sensor value
        # Add the new combined data back into the dictionary
        stats_df[f"{sensor}_stats"] = combine_sensor_stats(df_list=temp, prefix=sensor, stat_names=stat_names)
        
    print("Stats combined...")

    # Combine all the statistical data into one dataframe
    statistical_df_list = []
    for sensor in stats_df:
        statistical_df_list.append(stats_df[sensor])
    
    # Final statistical_df
    statistical_df = pd.concat(statistical_df_list, axis=1)
    print("Statistics made...\n", f"Size: {statistical_df.columns.size}")

    # Copy the statistical_df for the shake of the calculations
    copy_data = statistical_df.copy()
    
    # Coefficient of variance computation
    cv_df = copy_data.apply(lambda row: compute_coefficient_variance(row, sensors_for_statistics), axis=1, result_type='expand')
    print("Coefficient of Variance computed...\n", f"Size: {cv_df.columns.size}")

    # Stability ratio computation
    # Apply the function row by row
    stability_df = copy_data.apply(lambda row: compute_stability(row, sensors_for_statistics), axis=1, result_type='expand')
    print("Stability computed...\n", f"Size: {stability_df.columns.size}")

    # EMA computation
    ema_span = 5 #το 5 μπηκε αυθαιρετα

    for col in statistical_df.columns:
        ema_col_name = col + '_ema'
        statistical_df[ema_col_name] = statistical_df[col].ewm(span=ema_span, adjust=False).mean()

    # EMA dataframe
    ema_df = statistical_df[[col for col in statistical_df.columns if col.endswith('_ema')]]
    print("EMA computed...\n", f"Size: {ema_df.columns.size}")
    
    # Frequency analysis
    fft_dfs = {}
    for sensor in sensors:
        fft_dfs[sensor] = (fft_analysis(sensor_dict[sensor], sampling_rates_dict[sensor]), sampling_rates_dict[sensor])

    # Frequency
    band_power_df = pd.DataFrame()
    top_peaks_df = pd.DataFrame()
    spectral_df = pd.DataFrame()

    for sensor in sensors:

        freqs = get_freqs(fft_dfs[sensor][0], fft_dfs[sensor][1])

        # Band power
        band_power_cols = band_power(fft_dfs[sensor][0], freqs, [(0,10),(10, 50)], name=sensor)

        # Top N peaks
        top_peaks_cols = top_peaks(fft_dfs[sensor][0], freqs, name=sensor)

        # Spectral features
        spectral_cols = spectral_features(fft_dfs[sensor][0], freqs, name=sensor)

        # Saving into the dataframe
        band_power_df = pd.concat([band_power_df, band_power_cols], axis=1)
        top_peaks_df = pd.concat([top_peaks_df, top_peaks_cols], axis=1)
        spectral_df = pd.concat([spectral_df, spectral_cols], axis=1)

    # Frequency
    freq_df = pd.concat([band_power_df, top_peaks_df, spectral_df], axis=1)
    print("Frequency features computed...\n", f"Size: {freq_df.columns.size}")

    # Add all the dfs into one
    final_df = pd.concat([statistical_df, cv_df, stability_df, freq_df], axis=1)

    # Add profile_df if it isn't empty
    if profile_df.empty == False:
        profile_without_last_column = profile_df.iloc[:, :-1]
        final_with_profile_df = pd.concat([final_df, profile_without_last_column], axis=1)
    print("Profile features added...\n", f"Size:{final_with_profile_df.columns.size}")

    # Save the stable_flag column so it is not scaled
    stable_flag_nums = profile_df["stable_flag"]
    stable_flag_col = pd.DataFrame(stable_flag_nums, columns=["stable_flag", ])

    # Scale the data
    scaled_data = scaler.fit_transform(final_with_profile_df)
    scaled_data_dataset = pd.DataFrame(scaled_data, columns=final_with_profile_df.columns) # Just a columns argument needed
    final_scaled_df = pd.concat([scaled_data_dataset, stable_flag_col], axis=1)

    return final_scaled_df

# Calculate target data

# Helper functions

# Maintenance_score
def compute_maintenance_score(row, weights):
    # Compute an initial score using the provided weights
    base_score = (row['cooler_condition'] * weights['cooler_condition'] +
                  row['valve_condition'] * weights['valve_condition'] +
                  row['internal_pump_leakage'] * weights['internal_pump_leakage'] +
                  row['hydraulic_accumulator'] * weights['hydraulic_accumulator'])

    # If stable_flag is 0 (machine is stable), reduce the score
    if row['stable_flag'] == 0:
        return base_score * (1 + weights['stable_flag'])  # Apply stability weight

    return base_score  # Return the raw score without capping at 1

# Failure Mode Classification (multi-label flags)
def compute_failure_modes(df):
    failure_modes = pd.DataFrame()
    failure_modes["cooler_failure"] = (df["cooler_condition"] <= 20).astype(int)
    failure_modes["valve_failure"] = (df["valve_condition"] <= 80).astype(int)
    failure_modes["pump_failure"] = (df["internal_pump_leakage"] >= 1).astype(int)
    failure_modes["hydraulic_failure"] = (df["hydraulic_accumulator"] <= 100).astype(int)
    return failure_modes

# RUL = Time To Failure and we calculating it based on the failure_flag
def compute_ttf(failure_flags):
    ttf = np.zeros_like(failure_flags, dtype=int)
    next_failure = None
    for i in reversed(range(len(failure_flags))):
        if failure_flags[i]:
            next_failure = i
        ttf[i] = (next_failure - i) if next_failure is not None else len(failure_flags)
    return ttf

# Failure flag indicates when a machine is currently close to failing conditions
def create_failure_flag(row) -> bool:
  failure_flag = ((row["cooler_condition"] <= 3) |
  (row["valve_condition"] <= 70) |
  (row["internal_pump_leakage"] >= 2) |
  (row["hydraulic_accumulator"] <= 90)) and row["stable_flag"]

  return int(failure_flag)

# Health State Classification (multi-class) 
def compute_health_state(rul_series):
    values = [0, 1, 2]  # 0: Healthy, 1: Degraded, 2: Failing
    return pd.cut(rul_series, bins=[-1, 10, 30, float('inf')], labels=values[::-1], right=True).astype(int)

# Returns a dataframe with the target data the model can predict.
def target_data(training_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates target labels for predictive maintenance modeling based on the training_df and the profile_df calculated by process_data().

    Args:
        training_df (pd.DataFrame): DataFrame containing the sensor data and calculated statistical features.
        profile_df (pd.DataFrame): DataFrame containing machine condition labels (cooler, valve, pump leakage, accumulator pressure, and stability flag).

    Returns:
        pd.DataFrame: Target DataFrame containing:
            - Failure modes for each subsystem/component
            - Health state classification
            - Remaining Useful Life (RUL)
            - Failure flag
            - Maintenance score (normalized)

    Raises:
        ValueError: If the profile_df is empty.
    """

    if profile_df.empty:
        raise ValueError("The target dataframe cannot be formed without a profile dataframe.")

    # Maintenance_score
    columns_of_interest = [
        'cooler_condition',
        'valve_condition',
        'internal_pump_leakage',
        'hydraulic_accumulator',
        'stable_flag'
    ]

    # Calculating the correlation of all the features to the stable_flag
    correlation_matrix = profile_df[columns_of_interest].corr()

    # Calculating the weights of maintenance_score
    weights = correlation_matrix['stable_flag']
    weights_dict = weights.to_dict()  # Convert to dictionary for easier handling

    # Applying the computation of maintenance score for the current data
    training_df["maintenance_score"] = training_df.apply(lambda row: compute_maintenance_score(row, weights_dict), axis=1)

    # Scale maintenance_score to MinMax
    min_max_scaler = MinMaxScaler()
    training_df["maintenance_score"] = min_max_scaler.fit_transform(training_df[["maintenance_score"]])
    
    # Extracting maintenance_col as other dataframe
    maintenance_col = training_df["maintenance_score"]

    # Failure Flag
    failure_flag_col = pd.DataFrame(training_df.apply(lambda row: create_failure_flag(row), axis = 1), columns=["failure_flag", ])
    print(failure_flag_col)

    # RUL (Remaining Useful Life)
    rul_col = pd.DataFrame(compute_ttf(failure_flag_col.values), columns=["RUL", ])
    print(rul_col)

    # Health State
    health_state_col = pd.DataFrame(compute_health_state(rul_col["RUL"]), columns=["health_state", ])
    print(health_state_col)

    # Failure modes
    failure_modes_df = compute_failure_modes(training_df)
    print(failure_modes_df)

    # Putting the maintenance score to the end
    target_df = pd.concat([failure_flag_col, rul_col, health_state_col, failure_modes_df, maintenance_col], axis=1)
    return target_df

def extract_to_csv(data: pd.DataFrame, filename: str, file_extension: str = 'csv', separator: str = ',') -> None:
    """
    Extracts the given DataFrame into a CSV file with specified parameters.

    Args:
        data (pd.DataFrame): The DataFrame to be exported.
        filename (str): The name of the output file (without extension).
        file_extension (str, optional): The file extension to use (default is 'csv').
        separator (str, optional): The delimiter to use in the CSV file (default is ',').

    Returns:
        None

    Raises:
        ValueError: If the data is not a pandas DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The 'data' parameter must be a pandas DataFrame.")

    full_filename = f"{filename}.{file_extension}"
    data.to_csv(full_filename, sep=separator, index=False)
    print(f"Data successfully extracted to {full_filename}")

def import_target_data() -> pd.DataFrame:
    """
    Imports the target data from a CSV file named 'targets.csv'.

    Returns:
        pd.DataFrame: DataFrame containing the imported target data.

    Raises:
        FileNotFoundError: If the 'targets.csv' file does not exist.
    """
    try:
        df = pd.read_csv('targets.csv')
        print("Data successfully imported from targets.csv")
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError("The file 'targets.csv' was not found.") from e

# NOTE: To unload the data from a downloaded dataset run this file with the correct parameters. The following if statement will HAVE to 
# contain the information for your dataset
# Run Code
if __name__ == '__main__':  
    # Dataset parameters set
    
    # Path of the dataset
    folder_path = "data/condition+monitoring+of+hydraulic+systems"

    # Sensor names
    sensors = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",  # Pressure sensors
    "EPS1",  # Motor power
    "FS1", "FS2",  # Volume flow
    "TS1", "TS2", "TS3", "TS4",  # Temperature sensors
    "VS1",  # Vibration
    "CE",  # Cooling efficiency
    "CP",  # Cooling power
    "SE"  # Efficiency factor
    ]

    # Dictionary with sampling rates of the sensors
    sampling_rates_dict = {
    'PS1': 100, 'PS2': 100, 'PS3': 100, 'PS4': 100, 'PS5': 100, 'PS6': 100,
    'EPS1': 100, 'FS1': 10, 'FS2': 10,
    'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1,
    'VS1': 1,
    'CE': 1, 'CP': 1, 'SE': 1
    }

    # Statistic names
    stat_names = ["mean", "std", "min", "max", "range", "rms"]

    # List of sensors that appear more than once in the setup
    sensor_duplicates = {"PS":6, "TS":4, "FS":2}

    # Sensors list used for statistical analysis
    sensors_for_statistics = [
    "PS",  # Pressure sensors
    "EPS1",  # Motor power
    "FS",  # Volume flow
    "TS",  # Temperature sensors
    "VS1",  # Vibration
    "CE",  # Cooling efficiency
    "CP",  # Cooling power
    "SE"  # Efficiency factor
    ]

    # Profile
    has_profile_file = True
    profile_columns = ["cooler_condition", "valve_condition", "internal_pump_leakage", "hydraulic_accumulator", "stable_flag"]
    

    #^ TESTING load_data()
    #* DONE
    sensor_dfs_dict, profile_df = load_data(sensors=sensors, folder_path=folder_path, profile_columns=profile_columns, has_profile_file=has_profile_file)
    print("Data loaded from folder...")

    #^ TESTING process_data()
    #* DONE
    X_df = process_data(sensors=sensors, sensor_dict=sensor_dfs_dict, sensors_duplicates=sensor_duplicates,
                            stat_names=stat_names, sampling_rates_dict=sampling_rates_dict, sensors_for_statistics=sensors_for_statistics,
                            scaler=StandardScaler(), profile_df=profile_df)
    print(X_df)

    #^ TESTING target_data()
    targets_df = target_data(X_df, profile_df)
    print("X: ", X_df)
    print("Targets", targets_df)

    #^ Extract to csv
    extract_to_csv(data=X_df, filename="x", file_extension="csv", separator=",")
    extract_to_csv(data=targets_df, filename="targets", file_extension="csv", separator=",")



