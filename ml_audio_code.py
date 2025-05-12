import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Paths
audio_folder = 'AUD'
plots_folder = 'rec(2)'
recurrence_plots_folder = 'recurrence_plots'

# Create folders if they don't exist
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(recurrence_plots_folder, exist_ok=True)

# Parameters
duration_limit =10  # seconds
downsample_factor = 20
epsilon = 0.1  # threshold for recurrence

# Track failed files
failed_files = [] 

# Loop over each .wav file in the audio folder
for filename in os.listdir(audio_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(audio_folder, filename)

        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=None)

            # --- Plot and Save Waveform ---
            plt.figure(figsize=(14, 5))
            librosa.display.waveshow(y, sr=sr)
            plt.title(f"Waveform of {filename}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.tight_layout()

            # Save waveform plot
            waveform_save_path = os.path.join(plots_folder, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(waveform_save_path)
            plt.close()

            print(f" Saved waveform plot for {filename} -> {waveform_save_path}")

            # Normalize
            audio_time_series = (y - np.mean(y)) / np.std(y)

            # Crop
            max_samples = int(sr * duration_limit)
            if len(audio_time_series) < max_samples:
                print(f" Skipping {filename}: Audio too short ({len(audio_time_series)/sr:.2f} sec)")
                failed_files.append(filename)
                continue

            audio_time_series = audio_time_series[:max_samples]

            # Downsample
            audio_time_series = audio_time_series[::downsample_factor]

            print(f"Audio after cropping and downsampling: {len(audio_time_series)} samples")

            # Recurrence matrix
            distance_matrix = np.abs(audio_time_series[:, np.newaxis] - audio_time_series[np.newaxis, :])
            recurrence_matrix = (distance_matrix < epsilon).astype(float)

            # Save Recurrence Plot
            rp_save_path = os.path.join(recurrence_plots_folder, f"RP_{os.path.splitext(filename)[0]}.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(recurrence_matrix, cmap='binary', origin='lower')
            plt.title(f"Recurrence Plot for {filename}")
            plt.xlabel("Time (samples)")
            plt.ylabel("Time (samples)")
            plt.tight_layout()
            plt.savefig(rp_save_path)
            plt.close()

            print(f"Saved recurrence plot for {filename} -> {rp_save_path}")

        except Exception as e:
            print(f" Error processing {filename}: {e}")
            failed_files.append(filename)

# After loop
print("\n Processing done.")
print(f"Total failed files: {len(failed_files)}")
if failed_files:
    print("Failed files:")
    for f in failed_files:
        print(f" - {f}")