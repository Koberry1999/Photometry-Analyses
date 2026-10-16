# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd          # Data manipulation (DataFrames)
import tdt                   # Tucker-Davis Technologies (TDT) data reader
from scipy.signal import savgol_filter
import numpy as np           # Numerical operations
import os                    # File path operations
from scipy.signal import periodogram   # Power spectral density
from scipy import interpolate
import matplotlib.pyplot as plt        # Plotting
from datetime import datetime
import tkinter as tk                    # GUI for file dialogs
from matplotlib.lines import Line2D
from sklearn.model_selection import StratifiedShuffleSplit
from tkinter import filedialog          # Folder/file selection dialogs
from scipy.signal import detrend
import scipy
import itertools
from scipy.signal import butter, filtfilt   # Filtering (unused here)
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter
from scipy.stats import linregress      # Linear regression (for motion correction)
from scipy.optimize import curve_fit, minimize   # Curve fitting (for bleaching)
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import auc

# =============================================================================
# SETUP: GUI to ask user for a save directory
# =============================================================================
root = tk.Tk()
root.geometry("600x400")     # Set initial window size
root.resizable(True, True)
root.withdraw()              # Hide the root Tk window (we only want dialogs)
save_directory = filedialog.askdirectory(title='Select a save directory')


# =============================================================================
# FUNCTION: load_photometry
# Loads a TDT data block, extracts signal/control channels and event onsets,
# and returns DataFrames containing the photometry traces and event times.
# =============================================================================
def load_photometry():

    # Ask user to pick the folder containing TDT data
    folder_path = filedialog.askdirectory(title="Select a Foler for TDT Data")
    data = tdt.read_block(folder_path)

    # 465 nm channel = calcium-dependent signal (dopamine/GCaMP sensor)
    signal_channel = (data.streams._465A.data)
    # 405 nm channel = calcium-independent control (isosbestic point)
    control_channel = (data.streams._405A.data)

    # ---- Event onsets (in seconds from TDT epocs) ----
    trial_onset  = data.epocs.IR__.onset    # trial start TTL
    water_onset  = data.epocs.Rew_.onset    # reward/water delivery
    air_puff     = data.epocs.Puff.onset    # air puff (aversive)
    fake_solenoid = 0                       # placeholder (disabled)
    # right_events = data.epocs.Righ.onset  # unused

    # ---- Build time axis for the photometry samples ----
    # TDT sampling rate is ~1017 Hz; first sample at 0.00098304 s
    signal_channel_length = len(signal_channel)
    num_points = signal_channel_length
    start = (0.00098304)
    end = (num_points - 1) / 1017
    linspace_array = np.linspace(start, end, num=num_points)

    # ---- Build DataFrames for each event type ----
    trial_start_df   = pd.DataFrame()
    water_onset_df   = pd.DataFrame()
    air_puff_df      = pd.DataFrame()
    fake_solenoid_df = pd.DataFrame()

    # Subtract 2 s so that "Onset" marks the start of the pre-event baseline
    trial_start_df['Onset']   = trial_onset - 2
    water_onset_df['Onset']   = water_onset - 2
    air_puff_df['Onset']      = air_puff - 2
    fake_solenoid_df['Onset'] = fake_solenoid - 2


    # Label event types
    trial_start_df['Trial_Type']   = 'Trial Start'
    water_onset_df['Trial_Type']   = 'Water'
    air_puff_df['Trial_Type']      = 'Air Puff'
    fake_solenoid_df['Trial_Type'] = 'Fake Solenoid'

    # Numeric TTL codes: 1=Trial Start, 2=Water, 3=Air Puff, 4=Fake Solenoid
    trial_start_df['TTL']   = 1
    water_onset_df['TTL']   = 2
    air_puff_df['TTL']      = 3
    fake_solenoid_df['TTL'] = 4

    # Combine all events and sort chronologically
    all_events_df = pd.concat([trial_start_df, water_onset_df,
                               air_puff_df, fake_solenoid_df])
    all_events_df = all_events_df.sort_values(by='Onset').reset_index(drop=True)

    # -------------------------------------------------------------------------
    # clean_ttl_simple:
    # Keeps only valid trial sequences of the form [1, 3, 2] (trial -> air puff
    # -> water) or [1, 4, 2] (trial -> fake solenoid -> water). Discards any
    # incomplete or mis-ordered sequences.
    # -------------------------------------------------------------------------
    def clean_ttl_simple(df):
        ttl_list = df['TTL'].tolist()
        start_indices = [i for i, val in enumerate(ttl_list) if val == 1]
        valid_indices = []
        for start_idx in start_indices:
            if start_idx + 2 < len(ttl_list):
                if [ttl_list[start_idx], ttl_list[start_idx+1],
                    ttl_list[start_idx+2]] in [[1, 3, 2], [1, 4, 2]]:
                    valid_indices.extend([start_idx, start_idx+1, start_idx+2])
        valid_indices = sorted(set(valid_indices))
        return df.iloc[valid_indices].reset_index(drop=True)

    all_events_df = clean_ttl_simple(all_events_df)

    # Build a single DataFrame holding the raw photometry traces
    df = pd.DataFrame()
    df['Signal']    = signal_channel
    df['Control']   = control_channel
    df['TimeStamp'] = linspace_array

    return df, trial_start_df, water_onset_df, air_puff_df, \
           fake_solenoid_df, all_events_df


# =============================================================================
# FUNCTION: double_exponential
# Model used to fit slow bleaching decay:
#   const + amp_slow * exp(-t/tau_slow) + amp_fast * exp(-t/tau_fast)
# tau_fast = tau_slow * tau_multiplier
# =============================================================================
def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) \
                 + amp_fast * np.exp(-t / tau_fast)


# =============================================================================
# FUNCTION: find_delta_f
# Core preprocessing:
#   1. Discard the first ~2034 samples (initial artifact)
#   2. Fit double-exponential bleaching curve to signal AND control
#   3. Subtract the fit (detrend)
#   4. Regress control onto signal, subtract motion artifact
#   5. Compute dF/F (%) and z-score
#   6. Show diagnostic plots (fits, dF/F, PSD)
# =============================================================================
def find_delta_f(signal_df):
    # Skip first 2034 samples (about 2 s) - contains onset artifact
    seconds = signal_df['TimeStamp'].iloc[2034:].values
    signal  = signal_df['Signal'].iloc[2034:].values
    control = signal_df['Control'].iloc[2034:].values

    # (Optional smoothing - currently disabled)
    # window_size = 20
    # b = np.ones(window_size) / window_size
    # a = 1
    # signal  = filtfilt(b, a, signal)
    # control = filtfilt(b, a, control)

    # Estimate sampling rate: how many samples fall within 1 s
    sampling_rate = (seconds[seconds <= seconds[0] + 1].size - 1)

    # ---- Fit double exponential to signal (bleaching decay) ----
    max_sig = np.max(signal)
    init_params = [max_sig / 2, max_sig / 4, max_sig / 4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
    sig_params, _ = curve_fit(double_exponential, seconds, signal,
                              p0=init_params, bounds=bounds, maxfev=20000)
    signal_fit = double_exponential(seconds, *sig_params)

    # ---- Fit double exponential to control ----
    max_ctl = np.max(control)
    init_params_ctl = [max_ctl / 2, max_ctl / 4, max_ctl / 4, 3600, 0.1]
    bounds_ctl = ([0, 0, 0, 600, 0], [max_ctl, max_ctl, max_ctl, 36000, 1])
    ctl_params, _ = curve_fit(double_exponential, seconds, control,
                              p0=init_params_ctl, bounds=bounds_ctl,
                              maxfev=20000)
    control_fit = double_exponential(seconds, *ctl_params)

    # Subtract the fitted bleaching baseline
    signal_detrended  = signal  - signal_fit
    control_detrended = control - control_fit

    # ---- Motion correction: linear regression of control onto signal ----
    slope, intercept, r_value, _, _ = linregress(control_detrended,
                                                 signal_detrended)
    print('Regression Results:')
    print(f'Slope    : {slope:.3f}')
    print(f'Intercept: {intercept:.3f}')
    print(f'R-squared: {r_value ** 2:.3f}')

    # Estimate motion contribution from the control channel and subtract it
    est_motion       = intercept + slope * control_detrended
    signal_corrected = signal_detrended - est_motion
    scaled_control   = slope * control_detrended + intercept   # (unused)

    # ---- Compute dF/F and z-score ----
    delta_f_over_f         = signal_corrected / signal_fit
    delta_f_over_f_percent = 100 * signal_corrected / signal_fit
    signal_zscored = (delta_f_over_f - np.mean(delta_f_over_f)) \
                     / np.std(delta_f_over_f)

    # ---- Plot 1: raw traces with fitted baselines ----
    plt.figure(figsize=(12, 5), dpi=150)
    plt.plot(seconds, signal,  label='Raw Signal', alpha=0.6)
    plt.plot(seconds, signal_fit, label='Signal Fit')
    plt.plot(seconds, control, label='Raw Control', alpha=0.6)
    plt.plot(seconds, control_fit, label='Control Fit')
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence')
    plt.title('Double Exponential Baseline Fits')
    plt.legend()
    plt.show()

    # ---- Plot 2: dF/F % and z-scored signal ----
    plt.figure(figsize=(12, 6), dpi=150)
    plt.subplot(2, 1, 1)
    plt.plot(seconds, delta_f_over_f_percent, label='ΔF/F (%)')
    plt.xlabel('Time (s)')
    plt.ylabel('ΔF/F (%)')
    plt.title('ΔF/F Percent')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(seconds, signal_zscored, label='Z-Scored Signal', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Z-Score')
    plt.title('Z-Scored Signal')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 3: Power spectral density (check for residual slow drift) ----
    freqs, psd = periodogram(signal_zscored, fs=sampling_rate)
    plt.figure(figsize=(10, 5))
    plt.semilogx(freqs, psd, 'b', label='Signal PSD')
    plt.axvline(0.002, color='r', linestyle='--',
                label='0.002 hz cutoff ')
    plt.xlabel('Frequency (Hz)'); plt.ylabel('Power')
    plt.title('Power Spectrum: Check for Residual Bleaching/Slow Signals')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    return delta_f_over_f_percent


# =============================================================================
# FUNCTION: photometry_preprocessing
# Top-level pipeline:
#   1. Load TDT data
#   2. Run dF/F preprocessing
#   3. Save the dF/F trace and (optionally) raw calcium + event onsets
# =============================================================================
def photometry_preprocessing():
    df_photometry, trial_start_df, water_onset_df, air_puff_df, \
        fake_solenoid_df, all_events = load_photometry()

    # Compute dF/F (%) for the full recording
    signal_zscored = find_delta_f(df_photometry)

    # Build output DataFrame
    df = pd.DataFrame()
    df['TimeStamp'] = signal_zscored

    # Ask user for output filename
    filename = input('Please enter a name for z-scored photometry signal: ')
    path = os.path.join(save_directory, f"{filename}.csv")

    # Ask user whether to save raw calcium and event onsets
    save_ca     = int(input('Please enter a 1 if you want to save the raw calcium data: '))
    save_events = int(input('Please enter 1 if you want to save the event onset times: '))

    # Pre-build paths for all optional outputs
    path2 = os.path.join(save_directory, 'Raw calcium data.csv')
    path3 = os.path.join(save_directory, 'Trial_Start.csv')
    path4 = os.path.join(save_directory, 'Water_Onest.csv')   # (typo: "Onest")
    path5 = os.path.join(save_directory, 'Air_Puff.csv')
    path6 = os.path.join(save_directory, 'Fake_Solenoid.csv')
    path7 = os.path.join(save_directory, 'All Events.csv')

    # Always save the dF/F (%) trace
    df.to_csv(path, index=False)

    if save_ca == 1:
        df_photometry.to_csv(path2, index=False)

    if save_events == 1:
        trial_start_df.to_csv(path3, index=False)
        water_onset_df.to_csv(path4, index=False)
        air_puff_df.to_csv(path5, index=False)
        fake_solenoid_df.to_csv(path6, index=False)
        all_events.to_csv(path7, index=False)


# =============================================================================
# RUN
# =============================================================================
photometry_preprocessing()