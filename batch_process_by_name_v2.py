import os
import glob
import pandas as pd

# --- Configuration --- #
# The folders where your .out files are located
FOLDERS = [
    'Anchor displacement (MS1)',
    'Mooring line stiffness degradation (MS6)',
    'Broken mooring line (MS7)',
    'Flooded columns (F8)',
    'Marine growth on the platform (F11)'
]

# Map keywords in the FILENAME to the correct target label
LABEL_MAP = {
    'Healthy': 0,
    'A_': 1,      # Anchor displacement (using A_ to be specific)
    'ML2': 2,     # Mooring line stiffness degradation
    'B_M': 3,     # Broken mooring line
    'F8_BC': 4,   # Flooded column
    'MG': 5       # Marine growth
}

# The sensors we want to extract as predictors
SENSORS = [
    'Wind1VelX', 'PtfmSurge', 'PtfmHeave', 'PtfmPitch', 
    'TwrBsMxt', 'TwrBsMyt', 'FAIRTEN1', 'FAIRTEN2', 'FAIRTEN3',
    'ANCHTEN1', 'ANCHTEN2', 'ANCHTEN3', 'GenPwr', 'RotSpeed'
]

WINDOW_SIZE_SEC = 360 
OUTPUT_CSV = 'fowt_master_dataset.csv'

def process_file(file_path):
    filename = os.path.basename(file_path)
    label = None
    
    # 1. Check for 'Healthy' first (Baseline)
    if 'Healthy' in filename:
        label = 0
    else:
        # 2. Check for the other specific fault codes
        for keyword, l in LABEL_MAP.items():
            if keyword == 'Healthy': continue
            if keyword in filename:
                label = l
                break
            
    if label is None:
        print(f"  [SKIP] Could not determine label for {filename}")
        return []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            column_names = lines[6].strip().split()
        
        # Using raw string r'\s+' for the separator to avoid syntax warnings
        df_raw = pd.read_csv(file_path, sep=r'\s+', skiprows=8, names=column_names)
        
        if 'Time' not in df_raw.columns:
            df_raw.rename(columns={df_raw.columns[0]: 'Time'}, inplace=True)

        dt = df_raw['Time'].iloc[1] - df_raw['Time'].iloc[0]
        sampling_freq = 1.0 / dt
        rows_per_window = int(WINDOW_SIZE_SEC * sampling_freq)
        num_windows = len(df_raw) // rows_per_window
        
        samples = []
        for i in range(num_windows):
            start, end = i * rows_per_window, (i + 1) * rows_per_window
            window = df_raw.iloc[start:end]
            
            # Extract Mean and Std Dev for every sensor
            features = {f"{s}_Mean": window[s].mean() for s in SENSORS if s in window.columns}
            features.update({f"{s}_Std": window[s].std() for s in SENSORS if s in window.columns})
            features['Target'] = label
            samples.append(features)
        return samples

    except Exception as e:
        print(f"  [ERROR] {filename}: {e}")
        return []

def main():
    print("="*60)
    print(" FOWT BATCH PROCESSOR (Filename-based Labels)")
    print("="*60)
    
    all_data = []
    for folder in FOLDERS:
        if not os.path.exists(folder):
            print(f"[SKIP] Folder not found: '{folder}'")
            continue
            
        files = glob.glob(os.path.join(folder, "*.out"))
        print(f"[PROC] {folder}: Found {len(files)} files.")
        
        for f in files:
            file_samples = process_file(f)
            all_data.extend(file_samples)
            if file_samples:
                label = file_samples[0]['Target']
                print(f"    - {os.path.basename(f)} -> Labeled as {label}")

    if all_data:
        pd.DataFrame(all_data).to_csv(OUTPUT_CSV, index=False)
        print("\n" + "="*60)
        print(f" SUCCESS: {len(all_data)} samples saved to {OUTPUT_CSV}")
        print("="*60)

if __name__ == "__main__":
    main()
