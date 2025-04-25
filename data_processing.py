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
# The targets are not included in the returning dataframe
def process_data(sensors:list[str], sensor_dict:dict[str, pd.DataFrame], sensors_duplicates:dict[str, int], stat_names:list[str], sampling_rates_dict:dict[str, int], 
                 sensors_for_statistics:list[str], scaler:StandardScaler | MinMaxScaler, profile_df:pd.DataFrame, ):
    
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
        stats_df["{sensor}_stats"] = combine_sensor_stats(df_list=temp, prefix=sensor, stat_names=stat_names)
    
    # Combine all the statistical data into one dataframe
    statistical_df_list = []
    for sensor in stats_df:
        statistical_df_list.append(stats_df["{sensor}_stats"])
    
    # Final statistical_df
    statistical_df = pd.concat(statistical_df_list, axis=1)
    
    # Copy the statistical_df for the shake of the calculations
    copy_data = statistical_df.copy()

    # Coefficient of variance computation
    cv_df = copy_data.apply(lambda row: compute_coefficient_variance(row, sensors_for_statistics), axis=1, result_type='expand')
    
    # Stability ratio computation
    # Apply the function row by row
    stability_df = copy_data.apply(lambda row: compute_stability(row, sensors_for_statistics), axis=1, result_type='expand')
    
    # EMA computation
    ema_span = 5 #το 5 μπηκε αυθαιρετα

    for col in statistical_df.columns:
        ema_col_name = col + '_ema'
        statistical_df[ema_col_name] = statistical_df[col].ewm(span=ema_span, adjust=False).mean()

    # EMA dataframe
    ema_df = statistical_df[[col for col in statistical_df.columns if col.endswith('_ema')]]
    
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

    # Add all the dfs into one
    final_df = pd.concat([statistical_df, cv_df, stability_df, ema_df, freq_df, profile_df], axis=1)

    # Scale the data
    scaled_data = scaler.fit_transform(final_df)
    final_scaled_df = pd.DataFrame(scaled_data, columns=final_df.columns)

    return final_scaled_df


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
    

    # TESTING load_data()
    sensor_dfs_dict, profile_dict = load_data(sensors=sensors, folder_path=folder_path, profile_columns=profile_columns, has_profile_file=has_profile_file)
    print(sensor_dfs_dict, profile_dict)


# Plotting fft analysis
def plot_fft_spectrum(fft_df, instance_indices=[0], max_freq=None):
    """
    Plot FFT spectrum for selected instance(s).

    Parameters:
    - fft_df (pd.DataFrame): DataFrame of FFT magnitudes (rows = instances, columns = frequency bins).
    - instance_indices (list): which instance(s) to plot from the dataset.
    - max_freq (float): maximum frequency to show (e.g., 200 Hz).
    """
    # Extract frequency values from column names
    freqs = [float(col.replace('freq_', '').replace('Hz', '')) for col in fft_df.columns]

    plt.figure(figsize=(12, 6))

    for idx in instance_indices:
        plt.plot(freqs, fft_df.iloc[idx], label=f'Instance {idx}')

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Spectrum")
    if max_freq:
        plt.xlim(0, max_freq)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# # Creating a dictionary with keys the name and as values a tuple (df, sampling_rate)

# combined_data = pd.concat([combined_stats_ps, combined_stats_fs, combined_stats_ts, CE_stats, CP_stats, EPS1_stats, VS1_stats, SE_stats], axis=1)
# combined_data

# # the dataset for scaling
# profile_without_first_and_last_columns = profile.iloc[:, :-1]
# data_for_scalling = pd.concat([combined_data, profile_without_first_and_last_columns], axis=1)
# data_for_scalling

# from sklearn.preprocessing import MinMaxScaler
# scaler = StandardScaler()

# scaled_data = scaler.fit_transform(data_for_scalling)
# scaled_data_dtset = pd.DataFrame(scaled_data, columns=data_for_scalling.columns) # Just a columns argument needed
# scaled_data_dtset

# profile_last_column = profile.iloc[:, -1]
# total_data = pd.concat([scaled_data_dtset, profile_last_column], axis = 1)
# total_data



# ema_only

# # Coefficient of Variance
# cv_data = cv_df

# # Stability
# stability_data = stability_df

# # Set up the inserting points
# insert_at = total_data.shape[1] - 5
# df_first = total_data.iloc[:, :insert_at]
# df_last = total_data.iloc[:, insert_at:]

# training_table = pd.concat([df_first, freq_data, cv_data, stability_data, ema_only, df_last], axis=1)

# training_table

# columns_of_interest = [
#     'cooler_condition',
#     'valve_condition',
#     'internal_pump_leakage',
#     'hydraulic_accumulator',
#     'stable_flag'
# ]

# correlation_matrix = profile[columns_of_interest].corr()
# correlation_matrix

# import seaborn as sns

# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
# plt.title("Correlation Heatmap")
# plt.show()

# weights = correlation_matrix['stable_flag']

# weights_dict = weights.to_dict() #convert to dictionary for easier handling

# for feature, weight in weights_dict.items():
#     print(f"{feature}: {weight}")

# total = sum(abs(w) for w in weights_dict.values())
# normalized_weights = {k: abs(v) / total for k, v in weights_dict.items()}
# normalized_weights

# def compute_maintenance_score(row, weights):
#     # Compute an initial score using the provided weights
#     base_score = (row['cooler_condition'] * weights['cooler_condition'] +
#                   row['valve_condition'] * weights['valve_condition'] +
#                   row['internal_pump_leakage'] * weights['internal_pump_leakage'] +
#                   row['hydraulic_accumulator'] * weights['hydraulic_accumulator'])

#     # If stable_flag is 0 (machine is stable), reduce the score
#     if row['stable_flag'] == 0:
#         return base_score * (1 + weights['stable_flag'])  # Apply stability weight

#     return base_score  # Return the raw score without capping at 1

# training_table["maintenance_score"] = training_table.apply(lambda row: compute_maintenance_score(row, weights_dict), axis=1)
# # Scale maintenance_score to minMax
# min_max_scaler = MinMaxScaler()
# training_table["maintenance_score"] = min_max_scaler.fit_transform(training_table[["maintenance_score"]])
# training_table

# def create_failure_flag(row):

#   failure_flag = ((row["cooler_condition"] <= 3) |
#   (row["valve_condition"] <= 70) |
#   (row["internal_pump_leakage"] >= 2) |
#   (row["hydraulic_accumulator"] <= 90)) and row["stable_flag"]

#   return int(failure_flag)

# training_table["failure_flag"] = training_table.apply(lambda row: create_failure_flag(row), axis = 1)
# training_table

# # RUL = Time To Failure and we calculating it based on the failure_flag
# def compute_ttf(failure_flags):
#     ttf = np.zeros_like(failure_flags, dtype=int)
#     next_failure = None
#     for i in reversed(range(len(failure_flags))):
#         if failure_flags[i]:
#             next_failure = i
#         ttf[i] = (next_failure - i) if next_failure is not None else len(failure_flags)
#     return ttf

# training_table["rul"] = compute_ttf(training_table["failure_flag"].values)
# training_table

# # Health State Classification (multi-class)
# def compute_health_state(rul_series):
#     values = [0, 1, 2]  # 0: Healthy, 1: Degraded, 2: Failing
#     return pd.cut(rul_series, bins=[-1, 10, 30, float('inf')], labels=[2,1,0], right=True).astype(int)

# training_table["health_state"] = compute_health_state(training_table["rul"])
# training_table.health_state.value_counts()

# # Failure Mode Classification (multi-label flags)
# def compute_failure_modes(df):
#     failure_modes = pd.DataFrame()
#     failure_modes["cooler_failure"] = (df["cooler_condition"] <= 20).astype(int)
#     failure_modes["valve_failure"] = (df["valve_condition"] <= 80).astype(int)
#     failure_modes["pump_failure"] = (df["internal_pump_leakage"] >= 1).astype(int)
#     failure_modes["hydraulic_failure"] = (df["hydraulic_accumulator"] <= 100).astype(int)
#     return failure_modes

# failure_modes_df = compute_failure_modes(training_table)
# training_table = pd.concat([training_table, failure_modes_df], axis=1)
# training_table

# training_table

# list(training_table.columns)

# #i am placing maintenance_score last for easier handling later on the train test split
# maintenance_col = training_table['maintenance_score']

# training_table = training_table.drop(columns=['maintenance_score'])

# training_table['maintenance_score'] = maintenance_col

# X = training_table.drop(columns=['maintenance_score'])
# y = training_table['maintenance_score']

# from sklearn.model_selection import train_test_split

# X_train1, X_test1, y_train1, y_test1 = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# from sklearn.ensemble import RandomForestRegressor

# model = RandomForestRegressor(random_state=42)
# model.fit(X_train1, y_train1)


# importances = model.feature_importances_
# feature_names = X.columns
# feat_imp_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# # Visualizing
# plt.figure(figsize=(10, 6))
# plt.barh(feat_imp_df['Feature'][:20][::-1], feat_imp_df['Importance'][:20][::-1])
# plt.xlabel('Importance')
# plt.title('Top 20 Features for Maintenance Score')
# plt.tight_layout()
# plt.show()

# from sklearn.model_selection import train_test_split

# top_features = feat_imp_df['Feature'].head(20) # Top features from feature importance

# X_top = training_table[top_features]

# y = training_table['maintenance_score']

# X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)

# def get_score(model, X_train, X_test ,y_train, y_test):
#   model.fit(X_train, y_train)
#   return model.score(X_test, y_test)

# # importing the  models
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor

# rf_model = RandomForestRegressor(n_estimators=100)
# svm_model = SVR()
# knn_model = KNeighborsRegressor()

# print("rf score:" + str(get_score(rf_model, X_train, X_test, y_train, y_test)))
# print("svm score:" + str(get_score(svm_model, X_train, X_test, y_train, y_test)))
# print("kneighbohrs score:" + str(get_score(knn_model, X_train, X_test, y_train, y_test)))

# from sklearn.model_selection import KFold, cross_val_score

# kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# rf_scores = cross_val_score(rf_model, X_train, y_train, cv=kfold, scoring='r2')
# svm_scores = cross_val_score(svm_model, X_train, y_train, cv=kfold, scoring='r2')
# knn_scores = cross_val_score(knn_model, X_train, y_train, cv=kfold, scoring='r2')

# rf_scores

# svm_scores

# knn_scores

# def get_av(score_list):
#   av_score = np.median(score_list)
#   return av_score

# get_av(rf_scores) # average of rf

# get_av(svm_scores) # average of svm

# get_av(knn_scores) # average of knn

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.metrics import mean_squared_error, r2_score

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test) # no fit here because there will be data leakage

# dl_model = Sequential()
# dl_model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
# dl_model.add(Dense(64, activation='relu'))
# dl_model.add(Dense(32, activation='relu'))
# dl_model.add(Dense(1))


# dl_model.compile(optimizer='adam', loss='mse', metrics=['mae']) #compiling

# dl_model_fitted = dl_model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=16, verbose=1)

# import matplotlib.pyplot as plt

# plt.plot(dl_model_fitted.history['loss'], label='Training Loss')
# plt.plot(dl_model_fitted.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.title("Loss vs Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

# dl_model.evaluate(X_test_scaled, y_test)

# import matplotlib.pyplot as plt

# y_pred = dl_model.predict(X_test_scaled) # predictions

# # Graph
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Real Maintenance Score")
# plt.ylabel("Predicted Maintenance Score")
# plt.title("Predictions vs Real Values")
# plt.grid(True)
# plt.tight_layout()
# plt.show()



