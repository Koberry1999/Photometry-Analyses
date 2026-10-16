import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')  # headless-safe; remove if you want interactive plots
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
from collections import defaultdict
import os
from scipy import stats

# ============================================================================
# FILE PARSING AND LOADING FUNCTIONS
# ============================================================================

def parse_filename(filename):
    """Parse the filename to extract mouse ID and date"""
    name = Path(filename).stem
    parts = name.split('_')

    mouse_id = parts[0] if parts else "Unknown"

    date_pattern = r'(\d+)_(\d+)_(\d+)'
    date_match = re.search(date_pattern, name)

    if date_match:
        month = date_match.group(1)
        day = date_match.group(2)
        year = date_match.group(3)
        date = f"{month}_{day}_{year}"
    else:
        date_parts = []
        for i in range(len(parts)-1, -1, -1):
            if re.match(r'^\d+$', parts[i]):
                date_parts.insert(0, parts[i])
                if len(date_parts) == 3:
                    break
        date = '_'.join(date_parts) if date_parts else "Unknown_Date"

    return {
        'mouse_id': mouse_id,
        'date': date,
        'filename': filename
    }

def load_photometry_data(filepath):
    """Load RAW dF/F photometry data from CSV file (single column: TimeStamp)"""
    try:
        df = pd.read_csv(filepath)
        if 'TimeStamp' in df.columns:
            signal = df['TimeStamp'].values
        else:
            signal = df.iloc[:, 0].values
        print(f"  Loaded raw dF/F photometry with {len(signal)} samples")
        return signal
    except Exception as e:
        print(f"Error loading photometry {filepath}: {e}")
        return None

def load_events_data(filepath):
    """Load events data from CSV file with columns: Onset, Trial_Type, TTL"""
    try:
        df = pd.read_csv(filepath)
        print(f"  Loaded events with {len(df)} events")
        return df
    except Exception as e:
        print(f"Error loading events {filepath}: {e}")
        return None

def match_files(photometry_files, events_files):
    """Match photometry files with events files based on mouse ID and date"""
    matched_pairs = []

    for p_file in photometry_files:
        p_info = parse_filename(p_file)

        for e_file in events_files:
            e_info = parse_filename(e_file)

            if p_info['mouse_id'] == e_info['mouse_id'] and p_info['date'] == e_info['date']:
                matched_pairs.append({
                    'mouse_id': p_info['mouse_id'],
                    'date': p_info['date'],
                    'photometry_file': p_file,
                    'events_file': e_file
                })
                break

    return matched_pairs

# ============================================================================
# TRIAL COUNT SUMMARY FUNCTION
# ============================================================================

def save_trial_count_summary(matched_pairs, output_dir):
    summary_data = []

    print("\n" + "="*60)
    print("TRIAL COUNT SUMMARY (Raw Event Counts)")
    print("="*60)

    for pair in matched_pairs:
        mouse_id = pair['mouse_id']
        date = pair['date']
        events_df = load_events_data(pair['events_file'])

        if events_df is None:
            continue

        total_trials = len(events_df[events_df['TTL'] == 1])
        air_puff_events = len(events_df[events_df['TTL'] == 3])
        fake_solenoid_events = len(events_df[events_df['TTL'] == 4])
        water_events = len(events_df[events_df['TTL'] == 2])

        air_puff_complete = 0
        fake_solenoid_complete = 0

        trial_starts = events_df[events_df['TTL'] == 1].sort_values('Onset').reset_index(drop=True)

        for idx, start_row in trial_starts.iterrows():
            start_time = start_row['Onset']

            if idx + 1 < len(trial_starts):
                next_start = trial_starts.iloc[idx + 1]['Onset']
            else:
                next_start = start_time + 30

            events_between = events_df[
                (events_df['Onset'] > start_time) &
                (events_df['Onset'] < next_start)
            ].sort_values('Onset')

            has_air_puff = len(events_between[events_between['TTL'] == 3]) > 0
            has_water_after_air = False
            if has_air_puff:
                air_puff_time = events_between[events_between['TTL'] == 3].iloc[0]['Onset']
                water_after_air = events_between[
                    (events_between['TTL'] == 2) &
                    (events_between['Onset'] > air_puff_time)
                ]
                if len(water_after_air) > 0:
                    has_water_after_air = True

            has_fake_solenoid = len(events_between[events_between['TTL'] == 4]) > 0
            has_water_after_fake = False
            if has_fake_solenoid:
                fake_time = events_between[events_between['TTL'] == 4].iloc[0]['Onset']
                water_after_fake = events_between[
                    (events_between['TTL'] == 2) &
                    (events_between['Onset'] > fake_time)
                ]
                if len(water_after_fake) > 0:
                    has_water_after_fake = True

            if has_air_puff and has_water_after_air:
                air_puff_complete += 1
            if has_fake_solenoid and has_water_after_fake:
                fake_solenoid_complete += 1

        cleaned_air_puff = 0
        cleaned_fake_solenoid = 0

        ttl_list = events_df['TTL'].tolist()
        for i in range(len(ttl_list) - 2):
            if ttl_list[i] == 1 and ttl_list[i+1] == 3 and ttl_list[i+2] == 2:
                cleaned_air_puff += 1
            elif ttl_list[i] == 1 and ttl_list[i+1] == 4 and ttl_list[i+2] == 2:
                cleaned_fake_solenoid += 1

        summary_data.append({
            'Mouse_ID': mouse_id,
            'Date': date,
            'Total_Trial_Starts': total_trials,
            'Air_Puff_Events': air_puff_events,
            'Fake_Solenoid_Events': fake_solenoid_events,
            'Water_Events': water_events,
            'Complete_Air_Puff_Trials': air_puff_complete,
            'Complete_Fake_Solenoid_Trials': fake_solenoid_complete,
            'Cleaned_Air_Puff_Trials': cleaned_air_puff,
            'Cleaned_Fake_Solenoid_Trials': cleaned_fake_solenoid,
            'Total_Complete_Trials': air_puff_complete + fake_solenoid_complete,
            'Air_Puff_Percentage': (air_puff_complete / (air_puff_complete + fake_solenoid_complete) * 100) if (air_puff_complete + fake_solenoid_complete) > 0 else 0,
            'Fake_Solenoid_Percentage': (fake_solenoid_complete / (air_puff_complete + fake_solenoid_complete) * 100) if (air_puff_complete + fake_solenoid_complete) > 0 else 0
        })

        print(f"\nMouse: {mouse_id}, Date: {date}")
        print(f"  Total trial starts: {total_trials}")
        print(f"  Air Puff events: {air_puff_events}")
        print(f"  Fake Solenoid events: {fake_solenoid_events}")
        print(f"  Complete Air Puff trials (Start->Air Puff->Water): {air_puff_complete}")
        print(f"  Complete Fake Solenoid trials (Start->Fake->Water): {fake_solenoid_complete}")
        print(f"  Cleaned Air Puff trials: {cleaned_air_puff}")
        print(f"  Cleaned Fake Solenoid trials: {cleaned_fake_solenoid}")

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "Trial_Count_Summary_Raw_Events.csv", index=False)
    print(f"\nTrial count summary saved to: {output_dir / 'Trial_Count_Summary_Raw_Events.csv'}")

    mouse_day_summary = []
    for pair in matched_pairs:
        mouse_id = pair['mouse_id']
        date = pair['date']

        events_df = load_events_data(pair['events_file'])
        if events_df is None:
            continue

        total_trials = len(events_df[events_df['TTL'] == 1])
        air_puff_events = len(events_df[events_df['TTL'] == 3])
        fake_solenoid_events = len(events_df[events_df['TTL'] == 4])

        air_puff_complete = 0
        fake_solenoid_complete = 0
        trial_starts = events_df[events_df['TTL'] == 1].sort_values('Onset').reset_index(drop=True)

        for idx, start_row in trial_starts.iterrows():
            start_time = start_row['Onset']
            if idx + 1 < len(trial_starts):
                next_start = trial_starts.iloc[idx + 1]['Onset']
            else:
                next_start = start_time + 30

            events_between = events_df[
                (events_df['Onset'] > start_time) &
                (events_df['Onset'] < next_start)
            ].sort_values('Onset')

            has_air_puff = len(events_between[events_between['TTL'] == 3]) > 0
            has_water_after_air = False
            if has_air_puff:
                air_puff_time = events_between[events_between['TTL'] == 3].iloc[0]['Onset']
                water_after_air = events_between[
                    (events_between['TTL'] == 2) &
                    (events_between['Onset'] > air_puff_time)
                ]
                if len(water_after_air) > 0:
                    has_water_after_air = True

            has_fake_solenoid = len(events_between[events_between['TTL'] == 4]) > 0
            has_water_after_fake = False
            if has_fake_solenoid:
                fake_time = events_between[events_between['TTL'] == 4].iloc[0]['Onset']
                water_after_fake = events_between[
                    (events_between['TTL'] == 2) &
                    (events_between['Onset'] > fake_time)
                ]
                if len(water_after_fake) > 0:
                    has_water_after_fake = True

            if has_air_puff and has_water_after_air:
                air_puff_complete += 1
            if has_fake_solenoid and has_water_after_fake:
                fake_solenoid_complete += 1

        mouse_day_summary.append({
            'Mouse_ID': mouse_id,
            'Date': date,
            'Total_Trial_Starts': total_trials,
            'Air_Puff_Events': air_puff_events,
            'Fake_Solenoid_Events': fake_solenoid_events,
            'Complete_Air_Puff_Trials': air_puff_complete,
            'Complete_Fake_Solenoid_Trials': fake_solenoid_complete
        })

    mouse_day_df = pd.DataFrame(mouse_day_summary)
    mouse_day_df.to_csv(output_dir / "Trial_Count_By_Mouse_Day.csv", index=False)
    print(f"Per-mouse-day summary saved to: {output_dir / 'Trial_Count_By_Mouse_Day.csv'}")

    total_air = sum([d['Complete_Air_Puff_Trials'] for d in summary_data])
    total_fake = sum([d['Complete_Fake_Solenoid_Trials'] for d in summary_data])
    total_trials_all = total_air + total_fake

    print("\n" + "="*60)
    print("OVERALL TRIAL COUNT SUMMARY")
    print("="*60)
    print(f"Total Air Puff trials across all mice/days: {total_air}")
    print(f"Total Fake Solenoid trials across all mice/days: {total_fake}")
    print(f"Total trials across all mice/days: {total_trials_all}")
    if total_trials_all > 0:
        print(f"Air Puff percentage: {total_air/total_trials_all*100:.1f}%")
        print(f"Fake Solenoid percentage: {total_fake/total_trials_all*100:.1f}%")

    return summary_df, mouse_day_df

# ============================================================================
# EVENT TIMING FUNCTIONS
# ============================================================================

def calculate_event_timings(events_df):
    events_sorted = events_df.sort_values('Onset').reset_index(drop=True)

    timings = {
        'trial_start_to_water': [],
        'trial_start_to_air_puff': [],
        'trial_start_to_fake_solenoid': [],
        'air_puff_to_water': [],
        'fake_solenoid_to_water': []
    }

    trial_start_indices = events_sorted[events_sorted['TTL'] == 1].index.tolist()

    for i, start_idx in enumerate(trial_start_indices):
        start_time = events_sorted.iloc[start_idx]['Onset']

        if i + 1 < len(trial_start_indices):
            end_idx = trial_start_indices[i + 1]
        else:
            end_idx = len(events_sorted)

        events_in_trial = events_sorted.iloc[start_idx + 1:end_idx]

        if len(events_in_trial) == 0:
            continue

        water_events = events_in_trial[events_in_trial['TTL'] == 2]
        if len(water_events) > 0:
            water_time = water_events.iloc[0]['Onset']
            timings['trial_start_to_water'].append(water_time - start_time)

        air_puff_events = events_in_trial[events_in_trial['TTL'] == 3]
        if len(air_puff_events) > 0:
            air_puff_time = air_puff_events.iloc[0]['Onset']
            timings['trial_start_to_air_puff'].append(air_puff_time - start_time)

            water_after_air = events_in_trial[events_in_trial['Onset'] > air_puff_time]
            water_after_air = water_after_air[water_after_air['TTL'] == 2]
            if len(water_after_air) > 0:
                water_after_air_time = water_after_air.iloc[0]['Onset']
                timings['air_puff_to_water'].append(water_after_air_time - air_puff_time)

        fake_solenoid_events = events_in_trial[events_in_trial['TTL'] == 4]
        if len(fake_solenoid_events) > 0:
            fake_solenoid_time = fake_solenoid_events.iloc[0]['Onset']
            timings['trial_start_to_fake_solenoid'].append(fake_solenoid_time - start_time)

            water_after_fake = events_in_trial[events_in_trial['Onset'] > fake_solenoid_time]
            water_after_fake = water_after_fake[water_after_fake['TTL'] == 2]
            if len(water_after_fake) > 0:
                water_after_fake_time = water_after_fake.iloc[0]['Onset']
                timings['fake_solenoid_to_water'].append(water_after_fake_time - fake_solenoid_time)

    return timings

def plot_event_timings(matched_pairs, save_dir):
    animal_timings = defaultdict(lambda: {
        'trial_start_to_water': [],
        'trial_start_to_air_puff': [],
        'trial_start_to_fake_solenoid': [],
        'air_puff_to_water': [],
        'fake_solenoid_to_water': []
    })

    all_timings = {
        'trial_start_to_water': [],
        'trial_start_to_air_puff': [],
        'trial_start_to_fake_solenoid': [],
        'air_puff_to_water': [],
        'fake_solenoid_to_water': []
    }

    for pair in matched_pairs:
        mouse_id = pair['mouse_id']
        events_df = load_events_data(pair['events_file'])
        if events_df is None:
            continue

        timings = calculate_event_timings(events_df)

        for key, values in timings.items():
            if len(values) > 0:
                animal_timings[mouse_id][key].extend(values)
                all_timings[key].extend(values)

    has_data = any(len(v) > 0 for v in all_timings.values())
    if not has_data:
        print("No event timing data found to plot!")
        return

    timing_colors = {
        'trial_start_to_water': 'blue',
        'trial_start_to_air_puff': 'orange',
        'trial_start_to_fake_solenoid': 'green',
        'air_puff_to_water': 'red',
        'fake_solenoid_to_water': 'purple'
    }

    timing_labels = {
        'trial_start_to_water': 'Trial Start -> Water',
        'trial_start_to_air_puff': 'Trial Start -> Air Puff',
        'trial_start_to_fake_solenoid': 'Trial Start -> Fake Solenoid',
        'air_puff_to_water': 'Air Puff -> Water',
        'fake_solenoid_to_water': 'Fake Solenoid -> Water'
    }

    n_animals = len(animal_timings)
    if n_animals > 0:
        n_cols = min(2, n_animals)
        n_rows = (n_animals + n_cols - 1) // n_cols
        fig1, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_animals == 1:
            axes = [axes]
        axes_flat = [ax for row in axes for ax in row] if n_rows > 1 else axes

        for idx, (mouse_id, timings) in enumerate(sorted(animal_timings.items())):
            ax = axes_flat[idx] if idx < len(axes_flat) else None
            if ax is None:
                break

            x_positions = []
            y_positions = []
            y_errors = []
            labels = []
            colors = []

            for i, (key, values) in enumerate(timings.items()):
                if len(values) > 0:
                    x_positions.append(i)
                    y_positions.append(np.mean(values))
                    y_errors.append(np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0)
                    labels.append(timing_labels[key])
                    colors.append(timing_colors[key])

            if len(x_positions) > 0:
                ax.bar(x_positions, y_positions, yerr=y_errors, capsize=5,
                       color=colors, alpha=0.7, edgecolor='black')
                ax.set_xticks(x_positions)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Time (s)', fontsize=10)
                ax.set_title(f'Mouse {mouse_id} - Event Timings', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)

                for i, (key, values) in enumerate(timings.items()):
                    if len(values) > 0:
                        ax.text(i, y_positions[i] + y_errors[i] + 0.01,
                               f'n={len(values)}', ha='center', va='bottom', fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No timing data available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Mouse {mouse_id}', fontsize=12, fontweight='bold')

        for idx in range(len(animal_timings), len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()
        save_path1 = save_dir / "Event_Timings_By_Animal.png"
        plt.savefig(save_path1, dpi=300, bbox_inches='tight')
        print(f"\nEvent timings (by animal) saved to: {save_path1}")
        plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(12, 8))

    x_positions = []
    y_positions = []
    y_errors = []
    labels = []
    colors = []

    for i, (key, values) in enumerate(all_timings.items()):
        if len(values) > 0:
            x_positions.append(i)
            y_positions.append(np.mean(values))
            y_errors.append(np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0)
            labels.append(timing_labels[key])
            colors.append(timing_colors[key])

    if len(x_positions) > 0:
        ax.bar(x_positions, y_positions, yerr=y_errors, capsize=8,
               color=colors, alpha=0.7, edgecolor='black', width=0.6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=14)
        ax.set_title('Average Event Timings - Across All Animals', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        for i, (key, values) in enumerate(all_timings.items()):
            if len(values) > 0:
                ax.text(i, y_positions[i] + y_errors[i] + 0.01,
                       f'n={len(values)}', ha='center', va='bottom', fontsize=10)

        stats_text = "Timing Statistics (mean +/- SEM):\n"
        for i, (key, values) in enumerate(all_timings.items()):
            if len(values) > 0:
                stats_text += f"{timing_labels[key]}: {np.mean(values):.3f} +/- {np.std(values, ddof=1)/np.sqrt(len(values)):.3f}s (n={len(values)})\n"

        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path2 = save_dir / "Event_Timings_Average_Across_Animals.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Event timings (average across animals) saved to: {save_path2}")
    plt.close(fig2)

    timing_data = []
    for mouse_id, timings in animal_timings.items():
        for key, values in timings.items():
            for val in values:
                timing_data.append({
                    'Mouse_ID': mouse_id,
                    'Event_Type': timing_labels[key],
                    'Time_Seconds': val
                })

    timing_df = pd.DataFrame(timing_data)
    timing_df.to_csv(save_dir / "Event_Timing_Data.csv", index=False)

    return fig1, fig2

# ============================================================================
# ITI FUNCTIONS
# ============================================================================

def calculate_itis(events_df):
    trial_starts = events_df[events_df['TTL'] == 1].copy()
    trial_starts = trial_starts.sort_values('Onset').reset_index(drop=True)

    if len(trial_starts) < 2:
        return np.array([])

    itis = []
    for i in range(1, len(trial_starts)):
        iti = trial_starts.iloc[i]['Onset'] - trial_starts.iloc[i-1]['Onset']
        itis.append(iti)

    return np.array(itis)

def plot_iti_histograms(matched_pairs, events_files, save_dir):
    animal_itis = defaultdict(list)
    all_itis = []

    for pair in matched_pairs:
        mouse_id = pair['mouse_id']
        events_df = load_events_data(pair['events_file'])
        if events_df is None:
            continue

        itis = calculate_itis(events_df)
        if len(itis) > 0:
            animal_itis[mouse_id].extend(itis)
            all_itis.extend(itis)

    if len(all_itis) == 0:
        print("No ITIs found to plot!")
        return

    n_animals = len(animal_itis)
    fig1, axes = plt.subplots(n_animals + 1, 1, figsize=(12, 4 * (n_animals + 1)))

    for idx, (mouse_id, itis) in enumerate(sorted(animal_itis.items())):
        ax = axes[idx]
        ax.hist(itis, bins='auto', color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(itis), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(itis):.2f}s')
        ax.axvline(x=np.median(itis), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(itis):.2f}s')
        ax.set_xlabel('Inter-Trial Interval (s)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Mouse {mouse_id} - ITIs (n={len(itis)})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        stats_text = f'Mean: {np.mean(itis):.2f}s\nStd: {np.std(itis):.2f}s\nMin: {np.min(itis):.2f}s\nMax: {np.max(itis):.2f}s'
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax = axes[-1]
    ax.hist(all_itis, bins='auto', color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(all_itis), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_itis):.2f}s')
    ax.axvline(x=np.median(all_itis), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_itis):.2f}s')
    ax.set_xlabel('Inter-Trial Interval (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Combined Across All Animals - ITIs (n={len(all_itis)})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    stats_text = f'Mean: {np.mean(all_itis):.2f}s\nStd: {np.std(all_itis):.2f}s\nMin: {np.min(all_itis):.2f}s\nMax: {np.max(all_itis):.2f}s'
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path1 = save_dir / "ITI_Histograms_By_Animal.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"\nITI histograms (by animal) saved to: {save_path1}")
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(12, 8))

    ax.hist(all_itis, bins='auto', color='darkblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(all_itis), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_itis):.2f}s')
    ax.axvline(x=np.median(all_itis), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_itis):.2f}s')
    ax.set_xlabel('Inter-Trial Interval (s)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('All ITIs - Combined Across All Animals and Sessions', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    stats_text = f'Total ITIs: {len(all_itis)}\nMean: {np.mean(all_itis):.2f}s\nStd: {np.std(all_itis):.2f}s\nMin: {np.min(all_itis):.2f}s\nMax: {np.max(all_itis):.2f}s'
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path2 = save_dir / "ITI_Histograms_All_Trials.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"ITI histograms (all trials) saved to: {save_path2}")
    plt.close(fig2)

    iti_data = []
    for mouse_id, itis in animal_itis.items():
        for iti in itis:
            iti_data.append({'Mouse_ID': mouse_id, 'ITI': iti})

    iti_df = pd.DataFrame(iti_data)
    iti_df.to_csv(save_dir / "ITI_Data.csv", index=False)

    return fig1, fig2

# ============================================================================
# TRIAL EXTRACTION AND ALIGNMENT FUNCTIONS
# ============================================================================

def find_trial_and_water_for_type(events_df, trial_type, min_separation=0, debug=False):
    trial_starts = events_df[events_df['TTL'] == 1].copy()
    trial_starts = trial_starts.sort_values('Onset').reset_index(drop=True)

    if len(trial_starts) == 0:
        print("  No trial start events found!")
        return pd.DataFrame()

    if trial_type == 'Fake_Solenoid':
        target_ttl = 4
    else:
        target_ttl = 3

    trial_data = []
    skipped_no_target = 0    
    skipped_no_water = 0
    skipped_water_before_start = 0

    for idx, start_row in trial_starts.iterrows():
        start_time = start_row['Onset']

        if idx + 1 < len(trial_starts):
            next_start = trial_starts.iloc[idx + 1]['Onset']
        else:
            next_start = start_time + 30

        events_between = events_df[
            (events_df['Onset'] > start_time) &
            (events_df['Onset'] < next_start)
        ].sort_values('Onset')

        target_events = events_between[events_between['TTL'] == target_ttl]
        if len(target_events) == 0:
            skipped_no_target += 1
            continue

        target_time = target_events.iloc[0]['Onset']

        water_after = events_between[
            (events_between['TTL'] == 2) &
            (events_between['Onset'] > target_time)
        ]

        if len(water_after) > 0:
            water_time = water_after.iloc[0]['Onset']
            if start_time < water_time < next_start:
                trial_data.append({
                    'trial_start': start_time,
                    'target_time': target_time,
                    'water_time': water_time
                })
            else:
                skipped_water_before_start += 1
        else:
            skipped_no_water += 1

    if debug:
        print(f"  DEBUG {trial_type}:")
        print(f"    Total trials processed: {len(trial_starts)}")
        print(f"    Skipped (no target event): {skipped_no_target}")
        print(f"    Skipped (no water after target): {skipped_no_water}")
        print(f"    Skipped (water before trial start): {skipped_water_before_start}")
        print(f"    Complete trials found: {len(trial_data)}")

    if len(trial_data) == 0:
        print(f"  No complete trials found for {trial_type}!")
        return pd.DataFrame()

    trial_df = pd.DataFrame(trial_data)
    print(f"  Found {len(trial_df)} complete trials for {trial_type}")

    return trial_df

def extract_trials_aligned_to_water(photometry_signal, trial_df, epoch, sampling_rate=1017, debug=False):
    trial_data = []
    trial_start_times = []
    water_times = []
    skipped_out_of_bounds = 0

    for idx, row in trial_df.iterrows():
        water_time = row['water_time']
        trial_start_time = row['trial_start']

        before = int(water_time * sampling_rate - sampling_rate * epoch)
        after = int(water_time * sampling_rate + sampling_rate * epoch)

        before = max(0, before)
        after = min(len(photometry_signal), after)

        if after - before < 10:
            skipped_out_of_bounds += 1
            continue

        segment = photometry_signal[before:after]
        trial_data.append(segment)
        trial_start_times.append(trial_start_time)
        water_times.append(water_time)

    if debug:
        print(f"  DEBUG extraction: {len(trial_data)} trials extracted, {skipped_out_of_bounds} skipped (out of bounds)")

    if len(trial_data) == 0:
        return pd.DataFrame(), np.array([]), [], []

    max_len = max(len(t) for t in trial_data)
    padded = []
    for trial in trial_data:
        if len(trial) < max_len:
            pad = np.full(max_len - len(trial), np.nan)
            padded.append(np.concatenate([trial, pad]))
        else:
            padded.append(trial[:max_len])

    df = pd.DataFrame(padded)
    timepoints = np.linspace(-epoch, epoch, max_len)

    return df, timepoints, trial_start_times, water_times

def zscore_trials_by_water_baseline(individual_trials, timepoints,
                                    baseline_start_s=-2.1, baseline_end_s=-1.1,
                                    sampling_rate=1017, debug=False):
    """
    Z-score each trial using a baseline window relative to WATER DELIVERY.
    Baseline is pulled from the extracted trial snippet itself.

        baseline window = [water + baseline_start_s, water + baseline_end_s]
                        = [water - 2.1 s, water - 1.1 s]   (default)

        Z(t) = (z_raw(t) - mean(baseline)) / std(baseline)
    """
    n_trials = len(individual_trials)
    n_timepoints = len(timepoints)

    if n_timepoints > 1:
        dt = timepoints[1] - timepoints[0]
    else:
        dt = 1.0 / sampling_rate

    zero_idx = int(np.argmin(np.abs(timepoints)))     # index of water (t=0)

    def _idx(t_seconds):
        return int(round(zero_idx + t_seconds / dt))

    b_start = max(0, _idx(baseline_start_s))
    b_end   = max(0, min(n_timepoints, _idx(baseline_end_s) + 1))   # inclusive end

    if b_end - b_start < 2:
        print(f"  WARNING: water-relative baseline window [{baseline_start_s}, "
              f"{baseline_end_s}]s is outside the extracted trial. "
              f"Need epoch >= {abs(baseline_start_s)}. Aborting.")
        return individual_trials.copy() * np.nan, \
               np.full(n_trials, np.nan), np.full(n_trials, np.nan)

    zscored = individual_trials.copy()
    baseline_means = np.full(n_trials, np.nan)
    baseline_stds  = np.full(n_trials, np.nan)

    skipped_all_nan = 0
    skipped_zero_std = 0

    for trial_idx in range(n_trials):
        baseline_segment = individual_trials.iloc[trial_idx].values[b_start:b_end]

        if np.all(np.isnan(baseline_segment)):
            skipped_all_nan += 1
            zscored.iloc[trial_idx] = np.nan
            continue

        b_mean = float(np.nanmean(baseline_segment))
        b_std  = float(np.nanstd(baseline_segment, ddof=1)) \
                 if np.sum(~np.isnan(baseline_segment)) > 1 else 0.0

        baseline_means[trial_idx] = b_mean
        baseline_stds[trial_idx]  = b_std

        if b_std == 0 or not np.isfinite(b_std):
            skipped_zero_std += 1
            zscored.iloc[trial_idx] = np.nan
            continue

        zscored.iloc[trial_idx] = (individual_trials.iloc[trial_idx].values - b_mean) / b_std

    if debug:
        print(f"  DEBUG z-score (baseline = water {baseline_start_s:+.3f}s to "
              f"{baseline_end_s:+.3f}s):")
        print(f"    Trials z-scored: {n_trials - skipped_all_nan - skipped_zero_std}/{n_trials}")
        print(f"    Skipped (all-NaN baseline):        {skipped_all_nan}")
        print(f"    Skipped (zero/NaN baseline std):   {skipped_zero_std}")
        valid_means = baseline_means[~np.isnan(baseline_means)]
        valid_stds  = baseline_stds[~np.isnan(baseline_stds)]
        if len(valid_means) > 0:
            print(f"    Baseline mean across trials: mean={np.mean(valid_means):.4f}, "
                  f"std={np.std(valid_means):.4f}")
        if len(valid_stds) > 0:
            print(f"    Baseline std  across trials: mean={np.mean(valid_stds):.4f}, "
                  f"std={np.std(valid_stds):.4f}")

    return zscored, baseline_means, baseline_stds

def baseline_correct_trials_from_trial_start(individual_trials, timepoints, trial_start_times,
                                             water_times, photometry_signal, sampling_rate=1017, debug=False):
    """
    DEPRECATED — kept only for reference / easy revert.
    The new pipeline z-scores each trial using the trial-start baseline window
    (see zscore_trials_by_trial_start_baseline), so this old mean-subtraction step
    is no longer called from the main pipeline.
    """
    baseline_means = []
    skipped_no_baseline = 0

    for trial_idx in range(len(individual_trials)):
        water_time = water_times[trial_idx]

        baseline_start_idx = max(0, int((water_time - 1.2) * sampling_rate))
        baseline_end_idx = max(0, int((water_time-1) * sampling_rate))

        if baseline_start_idx >= baseline_end_idx:
            print(f"  Warning: Trial {trial_idx} has invalid baseline window indices")
            baseline_mean = 0.0
            skipped_no_baseline += 1
            baseline_means.append(baseline_mean)
            continue

        baseline_segment = photometry_signal[baseline_start_idx:baseline_end_idx]

        if len(baseline_segment) > 0:
            baseline_mean = np.mean(baseline_segment)
        else:
            print(f"  Warning: Trial {trial_idx} has empty baseline segment")
            baseline_mean = 0.0
            skipped_no_baseline += 1

        baseline_means.append(baseline_mean)

    baseline_corrected = individual_trials.copy()
    for i in range(len(baseline_means)):
        baseline_corrected.iloc[i] = individual_trials.iloc[i] - baseline_means[i]

    if debug:
        print(f"  DEBUG baseline correction: {len(baseline_corrected)} trials corrected, {skipped_no_baseline} with empty baseline")
        if len(baseline_means) > 0:
            print(f"  Baseline window: -1.2s to -1s before water delivery")
            print(f"  Baseline means: mean={np.mean(baseline_means):.4f}, std={np.std(baseline_means):.4f}")

    return baseline_corrected, np.array(baseline_means)

# ============================================================================
# CUSTOM CHANGE SCORE
# ============================================================================

def calculate_custom_change_score(individual_trials, timepoints, sampling_rate=1017,
                                  pre_start_s=-0.6, pre_end_s=-0.4,
                                  post_start_s=0.4, post_end_s=0.6,
                                  debug=False):
    n_timepoints = len(timepoints)
    if n_timepoints == 0 or len(individual_trials) == 0:
        return np.array([]), np.nan, np.nan, np.nan, np.nan

    if n_timepoints > 1:
        dt = timepoints[1] - timepoints[0]
    else:
        dt = 1.0 / sampling_rate

    zero_idx = int(np.argmin(np.abs(timepoints)))

    def _idx(t_seconds):
        return int(round(zero_idx + t_seconds / dt))

    pre_start_idx = max(0, _idx(pre_start_s))
    pre_end_idx   = max(0, min(n_timepoints, _idx(pre_end_s)))
    post_start_idx = max(0, min(n_timepoints, _idx(post_start_s)))
    post_end_idx   = max(0, min(n_timepoints, _idx(post_end_s)))

    if pre_end_idx <= pre_start_idx or post_end_idx <= post_start_idx:
        print(f"  WARNING: custom change-score window out of bounds "
              f"(pre {pre_start_idx}:{pre_end_idx}, post {post_start_idx}:{post_end_idx})")
        return np.zeros(len(individual_trials)), np.nan, np.nan, np.nan, np.nan

    trial_change_scores = []
    trial_pre_means = []
    trial_post_means = []

    for trial_idx in range(len(individual_trials)):
        trial = individual_trials.iloc[trial_idx].values
        pre_seg  = trial[pre_start_idx:pre_end_idx]
        post_seg = trial[post_start_idx:post_end_idx]

        pre_mean  = np.nanmean(pre_seg)  if not np.all(np.isnan(pre_seg))  else np.nan
        post_mean = np.nanmean(post_seg) if not np.all(np.isnan(post_seg)) else np.nan

        trial_pre_means.append(pre_mean)
        trial_post_means.append(post_mean)

        if np.isnan(pre_mean) or np.isnan(post_mean):
            trial_change_scores.append(np.nan)
        else:
            trial_change_scores.append(post_mean - pre_mean)

    trial_change_scores = np.array(trial_change_scores, dtype=float)
    valid_changes = trial_change_scores[~np.isnan(trial_change_scores)]

    avg_change = float(np.mean(valid_changes)) if len(valid_changes) > 0 else np.nan
    sem_change = float(np.std(valid_changes, ddof=1) / np.sqrt(len(valid_changes))) \
        if len(valid_changes) > 1 else 0.0

    valid_pre  = np.array([x for x in trial_pre_means  if not np.isnan(x)])
    valid_post = np.array([x for x in trial_post_means if not np.isnan(x)])

    if debug:
        print(f"  Custom change score window: "
              f"[{pre_start_s}, {pre_end_s}]s vs [{post_start_s}, {post_end_s}]s")
        if len(valid_pre) > 0:
            print(f"    Pre  mean across trials: {np.mean(valid_pre):.4f}  (n={len(valid_pre)})")
        if len(valid_post) > 0:
            print(f"    Post mean across trials: {np.mean(valid_post):.4f} (n={len(valid_post)})")
        print(f"    Change (post - pre):     {avg_change:.4f} +/- {sem_change:.4f} "
              f"(n_valid={len(valid_changes)}/{len(trial_change_scores)})")

    return trial_change_scores, avg_change, sem_change, \
           float(np.mean(valid_pre))  if len(valid_pre)  > 0 else np.nan, \
           float(np.mean(valid_post)) if len(valid_post) > 0 else np.nan

def calculate_window_auc_and_peak(individual_trials, timepoints, sampling_rate=1017,
                                  windows=None, debug=False):
    if windows is None:
        windows = [
            (-1.0, -0.5, 'm1_m05'),
            (-0.5,  0.0, 'm05_0'),
            ( 0.0,  0.75, '0_075'),
            ( 0.5,  1.0, '05_1'),
        ]

    n_trials = len(individual_trials)
    n_timepoints = len(timepoints)

    if n_trials == 0 or n_timepoints == 0:
        return {}

    dt = timepoints[1] - timepoints[0] if n_timepoints > 1 else 1.0 / sampling_rate
    zero_idx = int(np.argmin(np.abs(timepoints)))

    def _idx(t_seconds):
        return int(round(zero_idx + t_seconds / dt))

    results = {}

    for (start_s, end_s, label) in windows:
        start_idx = max(0, _idx(start_s))
        end_idx   = max(0, min(n_timepoints, _idx(end_s) + 1))

        if end_idx <= start_idx:
            print(f"  WARNING: window [{start_s}, {end_s}] out of bounds, skipping")
            continue

        trial_aucs = []
        trial_peaks = []

        for trial_idx in range(n_trials):
            segment = individual_trials.iloc[trial_idx].values[start_idx:end_idx]

            if np.all(np.isnan(segment)):
                auc = np.nan
            else:
                valid = ~np.isnan(segment)
                if valid.sum() < 2:
                    auc = np.nan
                else:
                    x_valid = np.arange(len(segment))[valid] * dt
                    y_valid = segment[valid]
                    auc = np.trapezoid(y_valid, x=x_valid)

            peak = np.nanmax(segment) if not np.all(np.isnan(segment)) else np.nan

            trial_aucs.append(auc)
            trial_peaks.append(peak)

        trial_aucs = np.array(trial_aucs, dtype=float)
        trial_peaks = np.array(trial_peaks, dtype=float)

        valid_aucs  = trial_aucs[~np.isnan(trial_aucs)]
        valid_peaks = trial_peaks[~np.isnan(trial_peaks)]

        results[f'AUC_{label}']  = trial_aucs
        results[f'Peak_{label}'] = trial_peaks

        results[f'AUC_{label}_mean'] = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else np.nan
        results[f'AUC_{label}_sem']  = float(np.std(valid_aucs, ddof=1) / np.sqrt(len(valid_aucs))) \
            if len(valid_aucs) > 1 else 0.0

        results[f'Peak_{label}_mean'] = float(np.mean(valid_peaks)) if len(valid_peaks) > 0 else np.nan
        results[f'Peak_{label}_sem']  = float(np.std(valid_peaks, ddof=1) / np.sqrt(len(valid_peaks))) \
            if len(valid_peaks) > 1 else 0.0

        results[f'AUC_{label}_n_valid']  = int(len(valid_aucs))
        results[f'Peak_{label}_n_valid'] = int(len(valid_peaks))

        if debug:
            print(f"  Window [{start_s:+.2f}, {end_s:+.2f}]s: "
                  f"AUC={results[f'AUC_{label}_mean']:.4f}+/-{results[f'AUC_{label}_sem']:.4f} "
                  f"(n_valid={len(valid_aucs)}/{n_trials}), "
                  f"Peak={results[f'Peak_{label}_mean']:.4f}+/-{results[f'Peak_{label}_sem']:.4f} "
                  f"(n_valid={len(valid_peaks)}/{n_trials})")

    return results

def align_water_trials_with_correct_baseline(photometry_file, events_file, epoch, trial_type,
                                             sampling_rate=1017, min_separation=0, debug=True,
                                             baseline_start_s=-2.1, baseline_end_s=-1.1):
    """
    Load RAW dF/F, extract water-aligned trials, and z-score each trial using a
    baseline window relative to WATER DELIVERY:

        baseline = [water - 2.1 s, water - 1.1 s]   (default)

        Z(t) = (dF/F(t) - mean(baseline)) / std(baseline)

    The returned `individual_trials` matrix is the per-trial z-scored trace (trials x timepoints).
    """
    photometry_signal = load_photometry_data(photometry_file)
    if photometry_signal is None:
        return None, None, None, None, None, None

    events_df = load_events_data(events_file)
    if events_df is None:
        return None, None, None, None, None, None

    trial_df = find_trial_and_water_for_type(events_df, trial_type, min_separation, debug=debug)
    if len(trial_df) == 0:
        return None, None, None, None, None, None

    individual_trials, timepoints, trial_start_times, water_times = extract_trials_aligned_to_water(
        photometry_signal, trial_df, epoch, sampling_rate, debug=debug
    )

    if len(individual_trials) == 0:
        return None, None, None, None, None, None

    # Per-trial z-score using the trial-start baseline window.
    zscored_trials, baseline_means, baseline_stds = zscore_trials_by_water_baseline(
        individual_trials, timepoints,
        baseline_start_s=baseline_start_s, baseline_end_s=baseline_end_s,
        sampling_rate=sampling_rate,
        debug=debug
    )

    averaged_trials = pd.DataFrame()
    averaged_trials['Mean'] = zscored_trials.mean(axis=0)
    averaged_trials['SEM'] = zscored_trials.sem(axis=0)
    averaged_trials['Timepoint'] = timepoints

    # "raw_individual_trials" now == the z-scored traces as well.
    # (There is no meaningful pre-z-score trace anymore: the z-score IS the normalization.)
    raw_individual_trials = zscored_trials.copy()
    raw_averaged_trials = pd.DataFrame()
    raw_averaged_trials['Mean'] = raw_individual_trials.mean(axis=0)
    raw_averaged_trials['SEM'] = raw_individual_trials.sem(axis=0)
    raw_averaged_trials['Timepoint'] = timepoints

    print(f"  Individual trials shape: {zscored_trials.shape} (rows=trials, cols=timepoints)")
    if len(baseline_means) > 0:
        print(f"  Per-trial z-score baseline window: water {baseline_start_s:+.3f}s to {baseline_end_s:+.3f}s")
        valid_means = baseline_means[~np.isnan(baseline_means)]
        if len(valid_means) > 0:
            print(f"  Baseline means: mean={np.mean(valid_means):.4f}, std={np.std(valid_means):.4f}")

    return zscored_trials, averaged_trials, baseline_means, \
           raw_individual_trials, raw_averaged_trials, trial_df

# ============================================================================
# PER-CONDITION PROCESSING HELPER
# ============================================================================

def process_single_condition(individual_trials, timepoints, raw_individual_trials,
                             mouse_id, date, day_number, trial_type,
                             sampling_rate, debug=True):
    avg_trace     = individual_trials.mean(axis=0).values
    sem_trace     = individual_trials.sem(axis=0).values
    raw_avg_trace = raw_individual_trials.mean(axis=0).values
    raw_sem_trace = raw_individual_trials.sem(axis=0).values

    n_trials = len(individual_trials)

    zero_idx = np.where(timepoints >= 0)[0][0] if len(np.where(timepoints >= 0)[0]) > 0 else len(timepoints)//2

    pre_start_idx  = max(0, zero_idx - int(0.25 * sampling_rate))
    pre_end_idx    = zero_idx
    post_start_idx = zero_idx
    post_end_idx   = min(len(timepoints), zero_idx + int(1 * sampling_rate))
    window_0_2_end = min(len(timepoints), zero_idx + int(2 * sampling_rate))

    pre_means, post_means, change_scores = [], [], []
    mean_z_scores, max_z_scores = [], []

    for trial_idx in range(len(individual_trials)):
        trial = individual_trials.iloc[trial_idx].values

        pre_seg  = trial[pre_start_idx:pre_end_idx]
        post_seg = trial[post_start_idx:post_end_idx]
        win_seg  = trial[zero_idx:window_0_2_end]

        pre_mean  = np.nanmean(pre_seg)  if not np.all(np.isnan(pre_seg))  else np.nan
        post_mean = np.nanmean(post_seg) if not np.all(np.isnan(post_seg)) else np.nan

        pre_means.append(pre_mean)
        post_means.append(post_mean)
        change_scores.append(
            post_mean - pre_mean if not (np.isnan(pre_mean) or np.isnan(post_mean)) else np.nan
        )

        if not np.all(np.isnan(win_seg)):
            mean_z_scores.append(np.nanmean(win_seg))
            max_z_scores.append(np.nanmax(win_seg))

    def _nanmean_sem(arr):
        arr = np.array(arr, dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan, np.nan
        m = float(np.mean(valid))
        s = float(np.std(valid, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0.0
        return m, s

    avg_change, sem_change = _nanmean_sem(change_scores)
    avg_mean_z, sem_mean_z = _nanmean_sem(mean_z_scores)
    avg_max_z,  sem_max_z  = _nanmean_sem(max_z_scores)

    _, custom_avg, custom_sem, custom_pre_mean, custom_post_mean = \
        calculate_custom_change_score(
            individual_trials, timepoints, sampling_rate,
            pre_start_s=-0.6, pre_end_s=-0.4,
            post_start_s=0.4, post_end_s=0.6,
            debug=debug
        )

    window_metrics = calculate_window_auc_and_peak(
        individual_trials, timepoints, sampling_rate,
        windows=[
            (-1.0, -0.5, 'm1_m05'),
            (-0.5,  0.0, 'm05_0'),
            ( 0.0,  0.75, '0_075'),
            ( 0.5,  1.0, '05_1'),
        ],
        debug=debug
    )

    window_summary = {k: v for k, v in window_metrics.items()
                      if k.endswith('_mean') or k.endswith('_sem') or k.endswith('_n_valid')}

    processed_dict = {
        'timepoints': timepoints,
        'avg_trace': avg_trace,
        'sem_trace': sem_trace,
        'raw_avg_trace': raw_avg_trace,
        'raw_sem_trace': raw_sem_trace,
        'mouse_id': mouse_id,
        'date': date,
        'day_number': day_number,
        'total_trials': n_trials,
        'avg_change_score': avg_change,
        'avg_mean_z': avg_mean_z,
        'avg_max_z': avg_max_z,
        'custom_change_score': custom_avg,
        'custom_sem_change': custom_sem,
        'custom_pre_mean': custom_pre_mean,
        'custom_post_mean': custom_post_mean,
        **window_summary
    }

    summary_row = {
        'Mouse_ID': mouse_id,
        'Day_Number': day_number,
        'Date': date,
        'Trial_Type': trial_type,
        'Total_Trials': n_trials,
        'Change_Score': avg_change,
        'SEM_Change': sem_change,
        'Mean_Z_Score_0_2s': avg_mean_z,
        'SEM_Mean_Z': sem_mean_z,
        'Max_Z_Score_0_2s': avg_max_z,
        'SEM_Max_Z': sem_max_z,
        'Custom_Change_Score': custom_avg,
        'SEM_Custom_Change': custom_sem,
        'Custom_Pre_Mean': custom_pre_mean,
        'Custom_Post_Mean': custom_post_mean,
        **window_summary
    }

    return processed_dict, summary_row

# ============================================================================
# PLOTTING FUNCTIONS - BASELINE CORRECTED (now: per-trial z-scored)
# ============================================================================

def plot_psth_by_mouse(processed_data, save_dir, epoch, trial_type_label, sample_rate=1017):
    mouse_data = {}
    for key, data in processed_data.items():
        mouse_id = data['mouse_id']
        if mouse_id not in mouse_data:
            mouse_data[mouse_id] = []
        mouse_data[mouse_id].append(data)

    n_mice = len(mouse_data)

    if n_mice == 0:
        print("No data to plot!")
        return

    fig, axes = plt.subplots(n_mice, 1, figsize=(12, 5 * n_mice))
    if n_mice == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for mouse_idx, (mouse_id, mouse_days) in enumerate(mouse_data.items()):
        ax = axes[mouse_idx]

        mouse_days_sorted = sorted(mouse_days, key=lambda x: x['day_number'])

        print(f"\nDEBUG PSTH plot - Mouse {mouse_id}, {trial_type_label}:")
        for day_idx, day_data in enumerate(mouse_days_sorted):
            color = colors[day_idx % len(colors)]

            timepoints = day_data['timepoints']
            avg_trace = day_data['avg_trace']
            sem_trace = day_data['sem_trace']
            n_trials = day_data['total_trials']

            print(f"  Day {day_data['day_number']}: n={n_trials} trials")

            time_seconds = timepoints

            ax.plot(time_seconds, avg_trace, linewidth=2.5,
                   color=color, label=f"Day {day_data['day_number']} (n={n_trials})")

            ax.fill_between(time_seconds,
                           avg_trace - sem_trace,
                           avg_trace + sem_trace,
                           alpha=0.25, color=color)

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Water Delivery')

        ax.set_xlabel('Time from Water Delivery (s)', fontsize=12)
        ax.set_ylabel('Z-score (per-trial, water baseline)', fontsize=12)
        ax.set_title(f'Mouse {mouse_id} - {trial_type_label} -> Water (Day-level traces)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-epoch, epoch])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    save_path = save_dir / f"PSTH_Plots_By_Mouse_Day_Level_{trial_type_label}_BaselineCorrected.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPSTH plot (day-level traces, per-trial z-scored) saved to: {save_path}")
    plt.close(fig)

    return fig

def plot_psth_average_across_animals_by_day(processed_data, save_dir, epoch, trial_type_label, sample_rate=1017):
    day_data = defaultdict(list)

    for key, data in processed_data.items():
        day_number = data['day_number']
        day_data[day_number].append(data)

    sorted_days = sorted(day_data.keys())
    n_days = len(sorted_days)

    if n_days == 0:
        print("No data to plot!")
        return

    n_cols = 3
    n_rows = (n_days + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else [axes]
    axes_flat = [ax for row in axes for ax in row] if n_rows > 1 else axes

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    print(f"\nDEBUG Average across animals - {trial_type_label}:")
    for day_idx, day_num in enumerate(sorted_days):
        ax = axes_flat[day_idx] if day_idx < len(axes_flat) else None
        if ax is None:
            break

        mouse_data_for_day = day_data[day_num]
        n_mice = len(mouse_data_for_day)

        print(f"  Day {day_num}: {n_mice} mice")

        for mouse_idx, mouse_data in enumerate(mouse_data_for_day):
            color = colors[mouse_idx % len(colors)]

            timepoints = mouse_data['timepoints']
            avg_trace = mouse_data['avg_trace']
            sem_trace = mouse_data['sem_trace']
            mouse_id = mouse_data['mouse_id']
            n_trials = mouse_data['total_trials']

            print(f"    Mouse {mouse_id}: n={n_trials} trials")

            time_seconds = timepoints

            ax.plot(time_seconds, avg_trace, linewidth=2,
                   color=color, label=f"Mouse {mouse_id} (n={n_trials})")

            ax.fill_between(time_seconds,
                           avg_trace - sem_trace,
                           avg_trace + sem_trace,
                           alpha=0.15, color=color)

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Water Delivery')

        ax.set_xlabel('Time from Water Delivery (s)', fontsize=10)
        ax.set_ylabel('Z-score (per-trial, water baseline)', fontsize=10)
        ax.set_title(f'Day {day_num} - {trial_type_label}->Water (n={n_mice} animals)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-epoch, epoch])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    for idx in range(len(sorted_days), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()

    save_path = save_dir / f"PSTH_Plots_Average_Across_Animals_By_Day_Individual_Mice_{trial_type_label}_BaselineCorrected.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPSTH plot (average across animals by day - individual mice shown, per-trial z-scored) saved to: {save_path}")
    plt.close(fig)

    return fig

def plot_psth_average_across_animals_combined(processed_data, save_dir, epoch, trial_type_label, sample_rate=1017):
    day_data = defaultdict(list)

    for key, data in processed_data.items():
        day_number = data['day_number']
        day_data[day_number].append(data)

    sorted_days = sorted(day_data.keys())
    n_days = len(sorted_days)

    if n_days == 0:
        print("No data to plot!")
        return

    n_cols = 3
    n_rows = (n_days + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else [axes]
    axes_flat = [ax for row in axes for ax in row] if n_rows > 1 else axes

    colors = plt.cm.viridis(np.linspace(0, 1, n_days))

    for day_idx, day_num in enumerate(sorted_days):
        ax = axes_flat[day_idx] if day_idx < len(axes_flat) else None
        if ax is None:
            break

        mouse_data_for_day = day_data[day_num]
        n_mice = len(mouse_data_for_day)

        min_len = min([len(data['avg_trace']) for data in mouse_data_for_day])

        mouse_traces = []
        for mouse_data in mouse_data_for_day:
            mouse_traces.append(mouse_data['avg_trace'][:min_len])

        mouse_traces = np.array(mouse_traces)
        avg_across_mice = np.mean(mouse_traces, axis=0)
        sem_across_mice = np.std(mouse_traces, axis=0, ddof=1) / np.sqrt(n_mice) if n_mice > 1 else np.zeros(mouse_traces.shape[1])

        timepoints = mouse_data_for_day[0]['timepoints'][:min_len]
        time_seconds = timepoints

        ax.plot(time_seconds, avg_across_mice, linewidth=3,
               color=colors[day_idx % len(colors)],
               label=f'Average across {n_mice} mice')

        ax.fill_between(time_seconds,
                       avg_across_mice - sem_across_mice,
                       avg_across_mice + sem_across_mice,
                       alpha=0.3, color=colors[day_idx % len(colors)])

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Water Delivery')

        ax.set_xlabel('Time from Water Delivery (s)', fontsize=10)
        ax.set_ylabel('Z-score (per-trial, water baseline)', fontsize=10)
        ax.set_title(f'Day {day_num} - {trial_type_label}->Water (n={n_mice} mice)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-epoch, epoch])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    for idx in range(len(sorted_days), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()

    save_path = save_dir / f"PSTH_Plots_Average_Across_Animals_By_Day_Combined_{trial_type_label}_BaselineCorrected.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPSTH plot (average across animals by day - combined trace, per-trial z-scored) saved to: {save_path}")
    plt.close(fig)

    return fig

# ============================================================================
# PLOTTING FUNCTIONS - RAW (now also per-trial z-scored)
# ============================================================================

def plot_psth_by_mouse_raw(processed_data_raw, save_dir, epoch, trial_type_label, sample_rate=1017):
    mouse_data = {}
    for key, data in processed_data_raw.items():
        mouse_id = data['mouse_id']
        if mouse_id not in mouse_data:
            mouse_data[mouse_id] = []
        mouse_data[mouse_id].append(data)

    n_mice = len(mouse_data)

    if n_mice == 0:
        print("No data to plot!")
        return

    fig, axes = plt.subplots(n_mice, 1, figsize=(12, 5 * n_mice))
    if n_mice == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for mouse_idx, (mouse_id, mouse_days) in enumerate(mouse_data.items()):
        ax = axes[mouse_idx]

        mouse_days_sorted = sorted(mouse_days, key=lambda x: x['day_number'])

        for day_idx, day_data in enumerate(mouse_days_sorted):
            color = colors[day_idx % len(colors)]

            timepoints = day_data['timepoints']
            avg_trace = day_data['avg_trace']
            sem_trace = day_data['sem_trace']
            n_trials = day_data['total_trials']

            time_seconds = timepoints

            ax.plot(time_seconds, avg_trace, linewidth=2.5,
                   color=color, label=f"Day {day_data['day_number']} (n={n_trials})")

            ax.fill_between(time_seconds,
                           avg_trace - sem_trace,
                           avg_trace + sem_trace,
                           alpha=0.25, color=color)

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Water Delivery')

        ax.set_xlabel('Time from Water Delivery (s)', fontsize=12)
        ax.set_ylabel('Z-score (per-trial, water baseline)', fontsize=12)
        ax.set_title(f'Mouse {mouse_id} - {trial_type_label} -> Water (Day-level traces)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-epoch, epoch])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    save_path = save_dir / f"PSTH_Plots_By_Mouse_Day_Level_{trial_type_label}_Raw.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPSTH plot (day-level traces) saved to: {save_path}")
    plt.close(fig)

    return fig

def plot_psth_average_across_animals_by_day_raw(processed_data_raw, save_dir, epoch, trial_type_label, sample_rate=1017):
    day_data = defaultdict(list)

    for key, data in processed_data_raw.items():
        day_number = data['day_number']
        day_data[day_number].append(data)

    sorted_days = sorted(day_data.keys())
    n_days = len(sorted_days)

    if n_days == 0:
        print("No data to plot!")
        return

    n_cols = 3
    n_rows = (n_days + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else [axes]
    axes_flat = [ax for row in axes for ax in row] if n_rows > 1 else axes

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for day_idx, day_num in enumerate(sorted_days):
        ax = axes_flat[day_idx] if day_idx < len(axes_flat) else None
        if ax is None:
            break

        mouse_data_for_day = day_data[day_num]
        n_mice = len(mouse_data_for_day)

        for mouse_idx, mouse_data in enumerate(mouse_data_for_day):
            color = colors[mouse_idx % len(colors)]

            timepoints = mouse_data['timepoints']
            avg_trace = mouse_data['avg_trace']
            sem_trace = mouse_data['sem_trace']
            mouse_id = mouse_data['mouse_id']
            n_trials = mouse_data['total_trials']

            time_seconds = timepoints

            ax.plot(time_seconds, avg_trace, linewidth=2,
                   color=color, label=f"Mouse {mouse_id} (n={n_trials})")

            ax.fill_between(time_seconds,
                           avg_trace - sem_trace,
                           avg_trace + sem_trace,
                           alpha=0.15, color=color)

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Water Delivery')

        ax.set_xlabel('Time from Water Delivery (s)', fontsize=10)
        ax.set_ylabel('Z-score (per-trial, water baseline)', fontsize=10)
        ax.set_title(f'Day {day_num} - {trial_type_label}->Water (n={n_mice} animals)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-epoch, epoch])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    for idx in range(len(sorted_days), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()

    save_path = save_dir / f"PSTH_Plots_Average_Across_Animals_By_Day_Individual_Mice_{trial_type_label}_Raw.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPSTH plot (average across animals by day - individual mice shown) saved to: {save_path}")
    plt.close(fig)

    return fig

def plot_psth_average_across_animals_combined_raw(processed_data_raw, save_dir, epoch, trial_type_label, sample_rate=1017):
    day_data = defaultdict(list)

    for key, data in processed_data_raw.items():
        day_number = data['day_number']
        day_data[day_number].append(data)

    sorted_days = sorted(day_data.keys())
    n_days = len(sorted_days)

    if n_days == 0:
        print("No data to plot!")
        return

    n_cols = 3
    n_rows = (n_days + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else [axes]
    axes_flat = [ax for row in axes for ax in row] if n_rows > 1 else axes

    colors = plt.cm.viridis(np.linspace(0, 1, n_days))

    for day_idx, day_num in enumerate(sorted_days):
        ax = axes_flat[day_idx] if day_idx < len(axes_flat) else None
        if ax is None:
            break

        mouse_data_for_day = day_data[day_num]
        n_mice = len(mouse_data_for_day)

        min_len = min([len(data['avg_trace']) for data in mouse_data_for_day])

        mouse_traces = []
        for mouse_data in mouse_data_for_day:
            mouse_traces.append(mouse_data['avg_trace'][:min_len])

        mouse_traces = np.array(mouse_traces)
        avg_across_mice = np.mean(mouse_traces, axis=0)
        sem_across_mice = np.std(mouse_traces, axis=0, ddof=1) / np.sqrt(n_mice) if n_mice > 1 else np.zeros(mouse_traces.shape[1])

        timepoints = mouse_data_for_day[0]['timepoints'][:min_len]
        time_seconds = timepoints

        ax.plot(time_seconds, avg_across_mice, linewidth=3,
               color=colors[day_idx % len(colors)],
               label=f'Average across {n_mice} mice')

        ax.fill_between(time_seconds,
                       avg_across_mice - sem_across_mice,
                       avg_across_mice + sem_across_mice,
                       alpha=0.3, color=colors[day_idx % len(colors)])

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Water Delivery')

        ax.set_xlabel('Time from Water Delivery (s)', fontsize=10)
        ax.set_ylabel('Z-score (per-trial, water baseline)', fontsize=10)
        ax.set_title(f'Day {day_num} - {trial_type_label}->Water (n={n_mice} mice)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-epoch, epoch])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    for idx in range(len(sorted_days), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()

    save_path = save_dir / f"PSTH_Plots_Average_Across_Animals_By_Day_Combined_{trial_type_label}_Raw.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPSTH plot (average across animals by day - combined trace) saved to: {save_path}")
    plt.close(fig)

    return fig

def plot_summary_by_day(processed_data, save_dir, trial_type_label, suffix='BaselineCorrected'):
    day_summary = defaultdict(lambda: {'change_scores': [], 'mean_z_scores': [], 'max_z_scores': [], 'n_animals': 0})

    for key, data in processed_data.items():
        day_number = data['day_number']
        day_summary[day_number]['change_scores'].append(data['avg_change_score'])
        day_summary[day_number]['mean_z_scores'].append(data['avg_mean_z'])
        day_summary[day_number]['max_z_scores'].append(data['avg_max_z'])
        day_summary[day_number]['n_animals'] += 1

    sorted_days = sorted(day_summary.keys())

    change_data = []
    mean_z_data = []
    max_z_data = []

    for day in sorted_days:
        change_data.append({
            'Day': day,
            'Mean_Change': np.mean(day_summary[day]['change_scores']),
            'SEM_Change': np.std(day_summary[day]['change_scores'], ddof=1) / np.sqrt(len(day_summary[day]['change_scores'])) if len(day_summary[day]['change_scores']) > 1 else 0,
            'N_Animals': day_summary[day]['n_animals']
        })
        mean_z_data.append({
            'Day': day,
            'Mean_Z': np.mean(day_summary[day]['mean_z_scores']),
            'SEM_Z': np.std(day_summary[day]['mean_z_scores'], ddof=1) / np.sqrt(len(day_summary[day]['mean_z_scores'])) if len(day_summary[day]['mean_z_scores']) > 1 else 0
        })
        max_z_data.append({
            'Day': day,
            'Max_Z': np.mean(day_summary[day]['max_z_scores']),
            'SEM_Max_Z': np.std(day_summary[day]['max_z_scores'], ddof=1) / np.sqrt(len(day_summary[day]['max_z_scores'])) if len(day_summary[day]['max_z_scores']) > 1 else 0
        })

    change_df = pd.DataFrame(change_data)
    mean_z_df = pd.DataFrame(mean_z_data)
    max_z_df = pd.DataFrame(max_z_data)

    change_df.to_csv(save_dir / f"Change_Score_Summary_By_Day_{trial_type_label}_{suffix}.csv", index=False)
    mean_z_df.to_csv(save_dir / f"Mean_Z_Score_Summary_By_Day_{trial_type_label}_{suffix}.csv", index=False)
    max_z_df.to_csv(save_dir / f"Max_Z_Score_Summary_By_Day_{trial_type_label}_{suffix}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.bar(change_df['Day'], change_df['Mean_Change'],
           yerr=change_df['SEM_Change'], capsize=5, color='steelblue',
           alpha=0.7, edgecolor='black')
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Change Score (0-1s - -250ms-0)', fontsize=12)
    ax.set_title(f'{trial_type_label}->Water: Change Score by Day ({suffix})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for i, row in change_df.iterrows():
        ax.text(row['Day'], row['Mean_Change'] + row['SEM_Change'] + 0.01,
               f"n={int(row['N_Animals'])}", ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    ax.bar(mean_z_df['Day'], mean_z_df['Mean_Z'],
           yerr=mean_z_df['SEM_Z'], capsize=5, color='forestgreen',
           alpha=0.7, edgecolor='black')
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Mean Z-score (0-2s)', fontsize=12)
    ax.set_title(f'{trial_type_label}->Water: Mean Z-score by Day ({suffix})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax = axes[2]
    ax.bar(max_z_df['Day'], max_z_df['Max_Z'],
           yerr=max_z_df['SEM_Max_Z'], capsize=5, color='coral',
           alpha=0.7, edgecolor='black')
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Max Z-score (0-2s)', fontsize=12)
    ax.set_title(f'{trial_type_label}->Water: Max Z-score by Day ({suffix})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    save_path = save_dir / f"Summary_Statistics_By_Day_{trial_type_label}_{suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSummary statistics plot saved to: {save_path}")
    plt.close(fig)

    return fig

# ============================================================================
# DAILY COMPARISON PLOTS (Air Puff vs Fake Solenoid)
# ============================================================================

def plot_per_mouse_per_day_comparison(processed_data_air, processed_data_fake, save_dir, epoch, sampling_rate=1017,subset_label='All'):
    mouse_day_data = defaultdict(lambda: {'air': [], 'fake': []})

    for key, data in processed_data_air.items():
        mouse_id = data['mouse_id']
        day_number = data['day_number']
        mouse_day_data[(mouse_id, day_number)]['air'].append(data)

    for key, data in processed_data_fake.items():
        mouse_id = data['mouse_id']
        day_number = data['day_number']
        mouse_day_data[(mouse_id, day_number)]['fake'].append(data)

    if not mouse_day_data:
        print("No data available for per-mouse-per-day comparison plots!")
        return

    for (mouse_id, day_num), data in sorted(mouse_day_data.items()):
        has_air = len(data['air']) > 0
        has_fake = len(data['fake']) > 0

        if not has_air and not has_fake:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if has_air:
            for mouse_data in data['air']:
                timepoints = mouse_data['timepoints']
                avg_trace = mouse_data['avg_trace']
                sem_trace = mouse_data['sem_trace']
                n_trials = mouse_data['total_trials']
                time_seconds = timepoints

                ax1.plot(time_seconds, avg_trace, linewidth=2.5,
                        color='blue', alpha=0.8,
                        label=f'Air Puff (n={n_trials})')
                ax1.fill_between(time_seconds,
                               avg_trace - sem_trace,
                               avg_trace + sem_trace,
                               alpha=0.2, color='blue')

        if has_fake:
            for mouse_data in data['fake']:
                timepoints = mouse_data['timepoints']
                avg_trace = mouse_data['avg_trace']
                sem_trace = mouse_data['sem_trace']
                n_trials = mouse_data['total_trials']
                time_seconds = timepoints

                ax1.plot(time_seconds, avg_trace, linewidth=2.5,
                        color='orange', alpha=0.8,
                        label=f'Fake Solenoid (n={n_trials})')
                ax1.fill_between(time_seconds,
                               avg_trace - sem_trace,
                               avg_trace + sem_trace,
                               alpha=0.2, color='orange')

        ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Water Delivery')
        ax1.set_ylim(-1.5, 4)
        ax1.set_xlabel('Time from Water Delivery (s)', fontsize=12)
        ax1.set_ylabel('Z-score (per-trial, water baseline)', fontsize=12)
        ax1.set_title(f'Mouse {mouse_id} - Day {day_num}', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.set_xlim([-epoch, epoch])

        # Right panel: same z-scored data (there is no separate "raw" trace anymore)
        if has_air:
            for mouse_data in data['air']:
                timepoints = mouse_data['timepoints']
                avg_trace = mouse_data['raw_avg_trace']
                sem_trace = mouse_data['raw_sem_trace']
                n_trials = mouse_data['total_trials']
                time_seconds = timepoints

                ax2.plot(time_seconds, avg_trace, linewidth=2.5,
                        color='blue', alpha=0.8,
                        label=f'Air Puff (n={n_trials})')
                ax2.fill_between(time_seconds,
                               avg_trace - sem_trace,
                               avg_trace + sem_trace,
                               alpha=0.2, color='blue')

        if has_fake:
            for mouse_data in data['fake']:
                timepoints = mouse_data['timepoints']
                avg_trace = mouse_data['raw_avg_trace']
                sem_trace = mouse_data['raw_sem_trace']
                n_trials = mouse_data['total_trials']
                time_seconds = timepoints

                ax2.plot(time_seconds, avg_trace, linewidth=2.5,
                        color='orange', alpha=0.8,
                        label=f'Fake Solenoid (n={n_trials})')
                ax2.fill_between(time_seconds,
                               avg_trace - sem_trace,
                               avg_trace + sem_trace,
                               alpha=0.2, color='orange')

        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Water Delivery')
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel('Time from Water Delivery (s)', fontsize=12)
        ax2.set_ylabel('Z-score (per-trial, water baseline)', fontsize=12)
        ax2.set_title(f'Mouse {mouse_id} - Day {day_num} (duplicate trace)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.set_xlim([-epoch, epoch])

        plt.tight_layout()

        save_path = save_dir / f"Mouse_{mouse_id}_Day_{day_num}_AirPuff_vs_FakeSolenoid_{subset_label}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Per-mouse-per-day plot saved to: {save_path}")
        plt.close(fig)

    print(f"\nGenerated {len(mouse_day_data)} per-mouse-per-day comparison plots")
    return mouse_day_data

def plot_daily_comparison_all_mice(processed_data_air, processed_data_fake, save_dir, epoch, sampling_rate=1017,subset_label='All'):
    day_data = defaultdict(lambda: {'air': [], 'fake': []})

    for key, data in processed_data_air.items():
        day_number = data['day_number']
        day_data[day_number]['air'].append(data)

    for key, data in processed_data_fake.items():
        day_number = data['day_number']
        day_data[day_number]['fake'].append(data)

    all_days = sorted(day_data.keys())

    if not all_days:
        print("No data available for daily comparison plots!")
        return

    mouse_colors = {}
    all_mice = set()
    for day in all_days:
        for data in day_data[day]['air']:
            all_mice.add(data['mouse_id'])
        for data in day_data[day]['fake']:
            all_mice.add(data['mouse_id'])

    color_palette = plt.cm.tab10(np.linspace(0, 1, len(all_mice)))
    for idx, mouse_id in enumerate(sorted(all_mice)):
        mouse_colors[mouse_id] = color_palette[idx]

    for day_num in all_days:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1, ax2 = axes.flatten()

        day_info = day_data[day_num]
        has_air = len(day_info['air']) > 0
        has_fake = len(day_info['fake']) > 0

        if has_air:
            for mouse_data in day_info['air']:
                mouse_id = mouse_data['mouse_id']
                color = mouse_colors[mouse_id]
                timepoints = mouse_data['timepoints']
                avg_trace = mouse_data['avg_trace']
                sem_trace = mouse_data['sem_trace']
                n_trials = mouse_data['total_trials']
                time_seconds = timepoints

                ax1.plot(time_seconds, avg_trace, linewidth=2,
                        color=color, alpha=0.8,
                        label=f'Mouse {mouse_id} (n={n_trials})')
                ax1.fill_between(time_seconds,
                               avg_trace - sem_trace,
                               avg_trace + sem_trace,
                               alpha=0.15, color=color)

        ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.set_ylim(-1.5, 4)
        ax1.set_xlabel('Time from Water (s)', fontsize=10)
        ax1.set_ylabel('Z-score', fontsize=10)
        ax1.set_title(f'Day {day_num}: Air Puff', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.set_xlim([-epoch, epoch])

        if has_fake:
            for mouse_data in day_info['fake']:
                mouse_id = mouse_data['mouse_id']
                color = mouse_colors[mouse_id]
                timepoints = mouse_data['timepoints']
                avg_trace = mouse_data['avg_trace']
                sem_trace = mouse_data['sem_trace']
                n_trials = mouse_data['total_trials']
                time_seconds = timepoints

                ax2.plot(time_seconds, avg_trace, linewidth=2,
                        color=color, alpha=0.8,
                        label=f'Mouse {mouse_id} (n={n_trials})')
                ax2.fill_between(time_seconds,
                               avg_trace - sem_trace,
                               avg_trace + sem_trace,
                               alpha=0.15, color=color)

        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel('Time from Water (s)', fontsize=10)
        ax2.set_ylabel('Z-score', fontsize=10)
        ax2.set_title(f'Day {day_num}: Fake Solenoid', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.set_xlim([-epoch, epoch])

        plt.tight_layout()
        save_path = save_dir / f"Day_{day_num}_All_Mice_AirPuff_vs_FakeSolenoid_{subset_label}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  All mice comparison plot saved to: {save_path}")
        plt.close(fig)

    print(f"\nGenerated {len(all_days)} all-mice comparison plots")

def plot_daily_comparison_average_across_mice(processed_data_air, processed_data_fake, save_dir, epoch, sampling_rate=1017,subset_label='All'):
    day_data = defaultdict(lambda: {'air': [], 'fake': []})

    for key, data in processed_data_air.items():
        day_number = data['day_number']
        day_data[day_number]['air'].append(data)

    for key, data in processed_data_fake.items():
        day_number = data['day_number']
        day_data[day_number]['fake'].append(data)

    all_days = sorted(day_data.keys())

    if not all_days:
        print("No data available for average comparison plots!")
        return

    for day_num in all_days:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

        day_info = day_data[day_num]
        has_air = len(day_info['air']) > 0
        has_fake = len(day_info['fake']) > 0

        if has_air:
            air_traces = []
            for mouse_data in day_info['air']:
                air_traces.append(mouse_data['avg_trace'])
            if air_traces:
                air_traces = np.array(air_traces)
                air_avg = np.mean(air_traces, axis=0)
                air_sem = np.std(air_traces, axis=0, ddof=1) / np.sqrt(len(air_traces))
                timepoints = day_info['air'][0]['timepoints']
                time_seconds = timepoints
                n_mice = len(day_info['air'])
                n_trials = sum([d['total_trials'] for d in day_info['air']])

                ax1.plot(time_seconds, air_avg, linewidth=3,
                        color='blue', label=f'Air Puff (n={n_mice} mice, {n_trials} trials)')
                ax1.fill_between(time_seconds, air_avg - air_sem, air_avg + air_sem,
                               alpha=0.25, color='blue')

        if has_fake:
            fake_traces = []
            for mouse_data in day_info['fake']:
                fake_traces.append(mouse_data['avg_trace'])
            if fake_traces:
                fake_traces = np.array(fake_traces)
                fake_avg = np.mean(fake_traces, axis=0)
                fake_sem = np.std(fake_traces, axis=0, ddof=1) / np.sqrt(len(fake_traces))
                timepoints = day_info['fake'][0]['timepoints']
                time_seconds = timepoints
                n_mice = len(day_info['fake'])
                n_trials = sum([d['total_trials'] for d in day_info['fake']])

                ax1.plot(time_seconds, fake_avg, linewidth=3,
                        color='orange', label=f'Fake Solenoid (n={n_mice} mice, {n_trials} trials)')
                ax1.fill_between(time_seconds, fake_avg - fake_sem, fake_avg + fake_sem,
                               alpha=0.25, color='orange')

        ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Water Delivery')
        ax1.set_ylim(-1.5, 4)
        ax1.set_xlabel('Time from Water Delivery (s)', fontsize=12)
        ax1.set_ylabel('Z-score (per-trial, water baseline)', fontsize=12)
        ax1.set_title(f'Day {day_num}: Average Across Mice', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.set_xlim([-epoch, epoch])

        plt.tight_layout()
        save_path = save_dir / f"Day_{day_num}_Average_Across_Mice_AirPuff_vs_FakeSolenoid_{subset_label}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Average across mice plot saved to: {save_path}")
        plt.close(fig)

    print(f"\nGenerated {len(all_days)} average-across-mice comparison plots")

# ============================================================================
# SAVE WIDE-FORMAT CUSTOM CHANGE SCORE CSV
# ============================================================================

def save_pre_post_csv(all_processed_data, output_dir, subset_label='All'):
    rows = {}

    window_labels = {
        'm1_m05': '-1_to_-0.5',
        'm05_0':  '-0.5_to_0',
        '0_075':  '0_to_0.75',
        '05_1':   '0.5_to_1',
    }

    for trial_type, processed_by_subset in all_processed_data.items():
        if subset_label not in processed_by_subset:
            continue
        processed = processed_by_subset[subset_label]

        for key, data in processed.items():
            row_key = (data['mouse_id'], data['day_number'], data['date'])
            if row_key not in rows:
                rows[row_key] = {
                    'Mouse_ID':   data['mouse_id'],
                    'Day_Number': data['day_number'],
                    'Date':       data['date'],
                }

            if trial_type == 'Air_Puff':
                prefix = 'AirPuff'
            elif trial_type == 'Fake_Solenoid':
                prefix = 'FakeSolenoid'
            else:
                continue

            rows[row_key][f'{prefix}_Total_Trials'] = data.get('total_trials', np.nan)
            rows[row_key][f'{prefix}_Pre_Mean']     = data.get('custom_pre_mean', np.nan)
            rows[row_key][f'{prefix}_Post_Mean']    = data.get('custom_post_mean', np.nan)
            rows[row_key][f'{prefix}_Change']       = data.get('custom_change_score', np.nan)

            for win_label, readable in window_labels.items():
                rows[row_key][f'{prefix}_AUC_{readable}']    = data.get(f'AUC_{win_label}_mean', np.nan)
                rows[row_key][f'{prefix}_AUC_{readable}_n']  = data.get(f'AUC_{win_label}_n_valid', np.nan)
                rows[row_key][f'{prefix}_Peak_{readable}']   = data.get(f'Peak_{win_label}_mean', np.nan)
                rows[row_key][f'{prefix}_Peak_{readable}_n'] = data.get(f'Peak_{win_label}_n_valid', np.nan)

    df = pd.DataFrame(list(rows.values()))

    preferred_cols = [
        'Mouse_ID', 'Day_Number', 'Date',
        'AirPuff_Total_Trials',
        'AirPuff_Pre_Mean', 'AirPuff_Post_Mean', 'AirPuff_Change',
        'FakeSolenoid_Total_Trials',
        'FakeSolenoid_Pre_Mean', 'FakeSolenoid_Post_Mean', 'FakeSolenoid_Change',
    ]
    for prefix in ['AirPuff', 'FakeSolenoid']:
        for readable in ['-1_to_-0.5', '-0.5_to_0', '0_to_0.75', '0.5_to_1']:
            preferred_cols.append(f'{prefix}_AUC_{readable}')
            preferred_cols.append(f'{prefix}_AUC_{readable}_n')
            preferred_cols.append(f'{prefix}_Peak_{readable}')
            preferred_cols.append(f'{prefix}_Peak_{readable}_n')

    df = df[[c for c in preferred_cols if c in df.columns]]
    if not df.empty:
        df = df.sort_values(['Mouse_ID', 'Day_Number']).reset_index(drop=True)

    out_path = output_dir / f"Pre_Post_Change_By_Mouse_Day_{subset_label}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPre/Post/Change + AUC/Peak CSV saved to: {out_path}")
    print(df.to_string(index=False))
    return df

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    root = tk.Tk()
    root.withdraw()

    print("Please select your photometry files (RAW dF/F)...")
    photometry_files = filedialog.askopenfilenames(
        title="Select Photometry Files (raw dF/F)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not photometry_files:
        print("No photometry files selected. Exiting.")
        return

    print(f"\nSelected {len(photometry_files)} photometry files")

    print("\nPlease select your All Events files...")
    events_files = filedialog.askopenfilenames(
        title="Select All Events Files",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not events_files:
        print("No events files selected. Exiting.")
        return

    print(f"Selected {len(events_files)} events files")

    print("\nPlease select where to save the results...")
    save_dir = filedialog.askdirectory(title="Select Save Directory")
    if not save_dir:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)

    output_dir = save_dir / "Water_Delivery_Analysis"
    output_dir.mkdir(exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    epoch = float(input("\nEnter the amount of time before and after the event (in seconds): "))

    if epoch < 2.1:
        print(f"ERROR: epoch = {epoch} s < 2.1 s. The water-relative baseline "
              f"window [-2.1, -1.1] would fall outside the extracted trial, "
              f"producing all-NaN traces. Aborting.")
        return

    sampling_rate = 1017

    # Baseline window for per-trial z-scoring, relative to TRIAL START
    baseline_start_s = -0.3
    baseline_end_s   = -0.1

    print("\nMatching photometry files with events files...")
    matched_pairs = match_files(photometry_files, events_files)

    if not matched_pairs:
        print("No matching pairs found! Make sure files have matching mouse IDs and dates.")
        return

    print(f"Found {len(matched_pairs)} matched pairs")

    # ------------------------------------------------------------------
    # TRIAL COUNT SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SAVING TRIAL COUNT SUMMARY")
    print("="*60)
    summary_df, mouse_day_df = save_trial_count_summary(matched_pairs, output_dir)

    # ------------------------------------------------------------------
    # EVENT TIMINGS
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PLOTTING EVENT TIMINGS")
    print("="*60)
    plot_event_timings(matched_pairs, output_dir)

    # ------------------------------------------------------------------
    # ITI HISTOGRAMS
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PLOTTING ITI HISTOGRAMS")
    print("="*60)
    plot_iti_histograms(matched_pairs, events_files, output_dir)

    # ------------------------------------------------------------------
    # Collect all dates per mouse (chronological day numbering)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("FIRST PASS: Collecting all dates for each mouse")
    print("="*60)

    mouse_all_dates = defaultdict(list)
    for pair in matched_pairs:
        mouse_id = pair['mouse_id']
        date = pair['date']
        if date not in mouse_all_dates[mouse_id]:
            mouse_all_dates[mouse_id].append(date)

    mouse_day_assignment = {}
    for mouse_id, dates in mouse_all_dates.items():
        sorted_dates = sorted(dates)
        mouse_day_assignment[mouse_id] = {}
        for day_num, date in enumerate(sorted_dates, 1):
            mouse_day_assignment[mouse_id][date] = day_num
        print(f"  {mouse_id}: {len(sorted_dates)} total sessions, days 1-{len(sorted_dates)}")

    # ------------------------------------------------------------------
    # Process both trial types PER ANIMAL/DAY so we can split on the
    # combined chronological trial order.
    # ------------------------------------------------------------------
    trial_types = [
        ('Fake_Solenoid', 'FakeSolenoid'),
        ('Air_Puff', 'AirPuff')
    ]

    subset_labels = ['All', 'FirstThird', 'LastThird']

    processed_data     = defaultdict(dict)
    processed_data_raw = defaultdict(dict)

    all_processed_data     = defaultdict(lambda: defaultdict(dict))
    all_processed_data_raw = defaultdict(lambda: defaultdict(dict))

    summary_rows = []

    for pair in matched_pairs:
        mouse_id = pair['mouse_id']
        date = pair['date']
        day_number = mouse_day_assignment.get(mouse_id, {}).get(date, 0)

        print("\n" + "="*60)
        print(f"Processing: {mouse_id} - Date: {date} (Day {day_number})")
        print("="*60)

        # ---- 1) Load + align BOTH trial types for this session ----
        session = {}   # trial_type -> dict of aligned data

        for trial_type, label_short in trial_types:
            print(f"\n  --- {trial_type} ---")
            result = align_water_trials_with_correct_baseline(
                pair['photometry_file'],
                pair['events_file'],
                epoch,
                trial_type,
                sampling_rate,
                min_separation=0,
                debug=True,
                baseline_start_s=baseline_start_s,
                baseline_end_s=baseline_end_s,
            )
            individual_trials, averaged_trials, baseline_means, \
                raw_individual_trials, raw_averaged_trials, trial_df = result

            if individual_trials is None or averaged_trials is None:
                print(f"  No {trial_type}->water trials found for {mouse_id} - Day {day_number}")
                continue

            # Save full-trial data
            individual_trials.to_csv(
                output_dir / f"{mouse_id}_Day{day_number}_{date}_{label_short}_Water_individual_trials_baseline_corrected.csv",
                index=False
            )
            averaged_trials.to_csv(
                output_dir / f"{mouse_id}_Day{day_number}_{date}_{label_short}_Water_averaged_baseline_corrected.csv",
                index=False
            )
            raw_individual_trials.to_csv(
                output_dir / f"{mouse_id}_Day{day_number}_{date}_{label_short}_Water_individual_trials_raw.csv",
                index=False
            )
            raw_averaged_trials.to_csv(
                output_dir / f"{mouse_id}_Day{day_number}_{date}_{label_short}_Water_averaged_raw.csv",
                index=False
            )

            if baseline_means is not None:
                # Rename to reflect that this is the per-trial baseline MEAN used for z-scoring.
                pd.DataFrame({
                    'Trial': range(len(baseline_means)),
                    'Baseline_Mean': baseline_means
                }).to_csv(
                    output_dir / f"{mouse_id}_Day{day_number}_{date}_{label_short}_Water_baseline_means.csv",
                    index=False
                )

            trial_df.to_csv(
                output_dir / f"{mouse_id}_Day{day_number}_{date}_{label_short}_Trial_Timings.csv",
                index=False
            )

            session[trial_type] = {
                'label_short': label_short,
                'individual': individual_trials,           # per-trial z-scored, rows = trials
                'raw_individual': raw_individual_trials,   # also z-scored (see note above)
                'timepoints': averaged_trials['Timepoint'].values,
                'trial_df': trial_df.reset_index(drop=True),
            }

        if not session:
            print(f"  No usable data for {mouse_id} - Day {day_number}. Skipping.")
            continue

        # ---- 2) Build the COMBINED chronological trial ordering ----
        combined_records = []
        for trial_type, s in session.items():
            tdf = s['trial_df']
            for local_idx, row in tdf.iterrows():
                combined_records.append((row['trial_start'], trial_type, local_idx))

        combined_records.sort(key=lambda x: x[0])
        n_combined = len(combined_records)

        combined_labels = np.full(n_combined, 'Middle', dtype=object)
        if n_combined > 0:
            b1 = int(np.floor(n_combined / 3.0))
            b2 = int(np.floor(2 * n_combined / 3.0))
            if n_combined == 1:
                b1, b2 = 1, 1
            elif n_combined == 2:
                b1, b2 = 1, 1
            if b1 == 0 and n_combined >= 3:
                b1 = 1
            if b2 <= b1 and n_combined >= 3:
                b2 = min(b1 + 1, n_combined)
            combined_labels[:b1] = 'FirstThird'
            combined_labels[b2:] = 'LastThird'

        labels_by_type = {
            tt: np.array(['Middle'] * len(s['trial_df']), dtype=object)
            for tt, s in session.items()
        }
        for global_idx, (_, trial_type, local_idx) in enumerate(combined_records):
            labels_by_type[trial_type][local_idx] = combined_labels[global_idx]

        print(f"\n  Combined trial count: {n_combined}")
        for lbl in ['FirstThird', 'Middle', 'LastThird']:
            n_lbl = int(np.sum(combined_labels == lbl))
            per_type = {tt: int(np.sum(labels_by_type[tt] == lbl)) for tt in session}
            print(f"    {lbl}: {n_lbl} combined  |  per type: {per_type}")

        # ---- 3) For each trial type, build All / FirstThird / LastThird subsets ----
        for trial_type, s in session.items():
            label_short = s['label_short']
            ind  = s['individual']
            rind = s['raw_individual']
            timepoints = s['timepoints']
            labels = labels_by_type[trial_type]

            mask_all   = np.ones(len(ind), dtype=bool)
            mask_first = (labels == 'FirstThird')
            mask_last  = (labels == 'LastThird')

            subsets = {
                'All':        (ind.iloc[mask_all],   rind.iloc[mask_all]),
                'FirstThird': (ind.iloc[mask_first], rind.iloc[mask_first]),
                'LastThird':  (ind.iloc[mask_last],  rind.iloc[mask_last]),
            }

            print(f"\n  {trial_type}: subsets -> " +
                  ", ".join(f"{k}={len(v[0])}" for k, v in subsets.items()))

            for subset_label, (ind_sub, raw_ind_sub) in subsets.items():
                if len(ind_sub) == 0:
                    print(f"    Skipping {subset_label} for {trial_type} (0 trials)")
                    continue

                data_dict, summary_row = process_single_condition(
                    ind_sub, timepoints, raw_ind_sub,
                    mouse_id, date, day_number, trial_type,
                    sampling_rate, debug=True
                )

                pd.DataFrame({
                    'Timepoint': timepoints,
                    'Mean': data_dict['avg_trace'],
                    'SEM':  data_dict['sem_trace'],
                }).to_csv(
                    output_dir / f"{mouse_id}_Day{day_number}_{date}_{label_short}_Water_{subset_label}_averaged_baseline_corrected.csv",
                    index=False
                )
                pd.DataFrame({
                    'Timepoint': timepoints,
                    'Mean': data_dict['raw_avg_trace'],
                    'SEM':  data_dict['raw_sem_trace'],
                }).to_csv(
                    output_dir / f"{mouse_id}_Day{day_number}_{date}_{label_short}_Water_{subset_label}_averaged_raw.csv",
                    index=False
                )

                key = f"{mouse_id}_{date}"

                all_processed_data[trial_type][subset_label][key] = data_dict
                all_processed_data_raw[trial_type][subset_label][key] = {
                    **data_dict,
                    'avg_trace': data_dict['raw_avg_trace'],
                    'sem_trace': data_dict['raw_sem_trace'],
                }

                processed_data[subset_label][f"{trial_type}__{key}"] = data_dict
                processed_data_raw[subset_label][f"{trial_type}__{key}"] = {
                    **data_dict,
                    'avg_trace': data_dict['raw_avg_trace'],
                    'sem_trace': data_dict['raw_sem_trace'],
                }

                summary_row['Subset'] = subset_label
                summary_rows.append(summary_row)

    # ------------------------------------------------------------------
    # After all pairs processed: save overall summaries
    # ------------------------------------------------------------------
    if summary_rows:
        summary_df_all = pd.DataFrame(summary_rows)
        summary_df_all.to_csv(output_dir / "Overall_Animal_Summary_AllSubsets.csv", index=False)
        print(f"\nCombined summary saved to: {output_dir / 'Overall_Animal_Summary_AllSubsets.csv'}")

        for trial_type, label_short in trial_types:
            for subset_label in subset_labels:
                sub = summary_df_all[
                    (summary_df_all['Trial_Type'] == trial_type) &
                    (summary_df_all['Subset'] == subset_label)
                ]
                if len(sub) == 0:
                    continue
                out_path = output_dir / f"Overall_Animal_Summary_{label_short}_{subset_label}.csv"
                sub.to_csv(out_path, index=False)
                print(f"Summary saved to: {out_path}")

    # ------------------------------------------------------------------
    # PLOTS: per (trial_type, subset)
    # ------------------------------------------------------------------
    for trial_type, label_short in trial_types:
        for subset_label in subset_labels:
            pd_sub = {
                k.split('__', 1)[1]: v
                for k, v in processed_data[subset_label].items()
                if k.startswith(f"{trial_type}__")
            }
            pd_raw_sub = {
                k.split('__', 1)[1]: v
                for k, v in processed_data_raw[subset_label].items()
                if k.startswith(f"{trial_type}__")
            }

            if not pd_sub:
                print(f"\nNo data for {trial_type} | {subset_label}; skipping plots.")
                continue

            tag = f"{trial_type}_{subset_label}"

            print(f"\n--- Plots for {trial_type} | {subset_label} ---")

            # PSTH - per-trial z-scored
            print(f"\nGenerating PSTH plots for {tag} (per-trial z-scored)...")
            plot_psth_by_mouse(pd_sub, output_dir, epoch, tag)
            plot_psth_average_across_animals_by_day(pd_sub, output_dir, epoch, tag)
            plot_psth_average_across_animals_combined(pd_sub, output_dir, epoch, tag)
            plot_summary_by_day(pd_sub, output_dir, tag, 'BaselineCorrected')

            # PSTH - duplicate (same z-scored data)
            print(f"\nGenerating PSTH plots for {tag} (duplicate trace)...")
            plot_psth_by_mouse_raw(pd_raw_sub, output_dir, epoch, tag)
            plot_psth_average_across_animals_by_day_raw(pd_raw_sub, output_dir, epoch, tag)
            plot_psth_average_across_animals_combined_raw(pd_raw_sub, output_dir, epoch, tag)
            plot_summary_by_day(pd_raw_sub, output_dir, tag, 'Raw')

    # ------------------------------------------------------------------
    # SAVE WIDE-FORMAT CUSTOM CHANGE SCORE CSV (one per subset)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SAVING CUSTOM CHANGE SCORE CSV")
    print("="*60)
    for subset_label in subset_labels:
        has_any = any(
            subset_label in all_processed_data[trial_type]
            for trial_type, _ in trial_types
        )
        if has_any:
            save_pre_post_csv(all_processed_data, output_dir, subset_label=subset_label)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"All results saved to: {output_dir}")
    print("="*60)

    # ------------------------------------------------------------------
    # DAILY COMPARISON PLOTS
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("GENERATING DAILY COMPARISON PLOTS (Air Puff vs Fake Solenoid)")
    print("="*60)

    if 'Air_Puff' in all_processed_data and 'Fake_Solenoid' in all_processed_data:
        for subset_label in subset_labels:
            air = all_processed_data['Air_Puff'].get(subset_label, {})
            fake = all_processed_data['Fake_Solenoid'].get(subset_label, {})
            if air and fake:
                print(f"\n--- Daily comparison plots | subset = {subset_label} ---")
                plot_per_mouse_per_day_comparison(
                    air, fake, output_dir, epoch, sampling_rate, subset_label=subset_label
                )
                plot_daily_comparison_all_mice(
                    air, fake, output_dir, epoch, sampling_rate, subset_label=subset_label
                )
                plot_daily_comparison_average_across_mice(
                    air, fake, output_dir, epoch, sampling_rate, subset_label=subset_label
                )
            else:
                print(f"Skipping daily comparison plots for subset '{subset_label}' - missing data")
    else:
        print("Skipping daily comparison plots - data not available")


if __name__ == "__main__":
    main()