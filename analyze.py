import yt_dlp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import simpleaudio as sa
import time
import threading
import argparse
from scipy import stats
import sounddevice as sd

# Define constants for BPM limits and frequency analysis
MIN_BPM_X = 80   # Minimum BPM for x-axis
MAX_BPM_X = 200  # Maximum BPM for x-axis
MIN_BPM_Y = 110  # Minimum BPM for y-axis
MAX_BPM_Y = 160  # Maximum BPM for y-axis
SAMPLE_RATE = 1000  # Sampling rate in Hz for better resolution

# Define the data directory relative to the script's location
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)  # Create the data directory if it doesn't exist

def download_youtube_audio(url, keep_original=False):
    print("Starting download...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Save as WAV for better compatibility with librosa
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(data_dir, '%(title)s.%(ext)s'),  # Save in data directory
    }

    info_dict = yt_dlp.YoutubeDL().extract_info(url, download=False)
    title = info_dict.get('title', None)
    audio_file_wav = os.path.join(data_dir, f"{title}.wav")
    audio_file_mp3 = os.path.join(data_dir, f"{title}.mp3")

    if os.path.exists(audio_file_wav):
        print(f"File '{audio_file_wav}' already exists. Skipping download.")
        return audio_file_wav  # Return existing WAV file name

    if os.path.exists(audio_file_mp3):
        print(f"File '{audio_file_mp3}' already exists. Skipping download.")
        return audio_file_mp3  # Return existing MP3 file name

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if keep_original:
        print(f"Keeping original downloaded files.")

    print("Download completed.")
    return audio_file_wav  # Return the filename of the downloaded audio

def analyze_audio(y, sr):
    print("Analyzing audio for BPM...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]  # Get the first element if it's an array

    if tempo > 0:
        print(f'Estimated BPM: {tempo:.2f}')
    else:
        print('No beats detected or unable to estimate BPM.')

    return beat_frames

def calculate_bpm(y, sr, beat_frames):
    """Calculate and return smoothed BPM values."""
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    bpm_values = []

    for i in range(len(beat_times) - 1):
        bpm_value = (60 / (beat_times[i + 1] - beat_times[i]))  # Calculate BPM between beats

        if bpm_value > 0 and MIN_BPM_Y <= bpm_value <= MAX_BPM_Y:  # Only consider BPM values within range
            if not bpm_values or bpm_value != bpm_values[-1]:  # Store only if it changes
                bpm_values.append(bpm_value)

    # Smooth the BPM values using a simple moving average or exponential smoothing
    if len(bpm_values) > 0:
        smoothed_bpm = np.convolve(bpm_values, np.ones(5)/5, mode='valid')  # Simple moving average
        return smoothed_bpm

    return np.array([])  # Return an empty array if no BPM values were calculated

def plot_waveform_and_bpm(y, sr, bpm_values, audio_file, mode_value):
    plt.figure(figsize=(12, 6))

    duration = len(y) / sr

    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')

    # Plot BPM over time on the same x-axis
    plt.subplot(2, 1, 2)

    plt.plot(np.linspace(0, duration, len(bpm_values)), bpm_values, color='r', label='BPM')

    plt.title('BPM Analysis')

    plt.xlabel('Time (MM:SS)')

    plt.xticks(ticks=np.arange(0, duration + 1, step=30),
               labels=[f"{int(i // 60):02}:{int(i % 60):02}" for i in np.arange(0, duration + 1, step=30)])

    plt.ylim(MIN_BPM_Y, MAX_BPM_Y)  # Set y-limits based on specified range

    plt.yticks(np.arange(MIN_BPM_Y, MAX_BPM_Y + 5, step=5))  # Set y-ticks every 5 BPM

    # Save plots as an image file with the name of the audio file + "_waveform_bpm_{mode_value}.jpg"
    plot_filename = os.path.join(data_dir, f"{os.path.splitext(audio_file)[0]}_waveform_bpm_{int(mode_value)}.jpg")
    plt.savefig(plot_filename, format='jpg')
    print(f"Plots saved as '{plot_filename}'.")

    plt.close()

def plot_fft(y, audio_file):
    """Plot FFT of the waveform focusing on low frequencies (converted to BPM)."""
    N = len(y)
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, d=1/SAMPLE_RATE)[:N // 2]

    plt.figure(figsize=(12, 6))

    # Convert frequencies to BPM (multiply by 60)
    bpm_values = xf * 60

    # Filter frequencies corresponding to desired BPM range (80-200 BPM)
    indices = np.where((bpm_values >= MIN_BPM_X) & (bpm_values <= MAX_BPM_X))

    plt.plot(bpm_values[indices], np.abs(yf[indices]), color='b')

    plt.title('Fourier Transform of Audio Waveform (Converted to BPM)')
    plt.xlabel('Frequency (BPM)')
    plt.ylabel('Amplitude')

    plt.xlim(MIN_BPM_X, MAX_BPM_X)   # Limit x-axis to show frequencies between MIN_BPM_X and MAX_BPM_X

    fft_filename = os.path.join(data_dir, f"{os.path.splitext(audio_file)[0]}_fft.jpg")
    plt.savefig(fft_filename)
    print(f"FFT plot saved as '{fft_filename}'.")

    plt.close()

def play_audio(file_path):
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    return play_obj

def print_bpm_during_playback(beat_times, duration):
    start_time = time.time()
    last_printed_bpm = None

    live_bpm_values = []

    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time

        window_start_time = elapsed_time - 5.0

        recent_beats = [bt for bt in beat_times if window_start_time < bt <= elapsed_time]

        current_time_seconds = int(elapsed_time)

        if len(recent_beats) > 1:
            bpm_count = len(recent_beats) - 1
            time_interval = recent_beats[-1] - recent_beats[0]

            if time_interval > 0:
                current_bpm = round((bpm_count / time_interval) * 60.0)
                if current_bpm != last_printed_bpm:
                    print(f'Current BPM at {current_time_seconds}s: {current_bpm} BPM')
                    last_printed_bpm = current_bpm
                    live_bpm_values.append(current_bpm)
        else:
            # Print every ten seconds even if there's no change
            if current_time_seconds % 10 == 0:
                print(f'Current BPM at {current_time_seconds}s: {last_printed_bpm} (no change)')

        time.sleep(1)

    return live_bpm_values

def save_actual_bpm_plot(bpm_values, audio_file):
    """Plot and save the actual BPM values during playback."""
    plt.figure(figsize=(12,6))

    plt.plot(bpm_values, color='g', label='Live BPM')

    plt.title('Live BPM Analysis')
    plt.xlabel('Time (Samples)')
    plt.ylabel('BPM')

    plt.ylim(MIN_BPM_Y - 10, MAX_BPM_Y + 10) # Allow some space above and below

    # Save plots as an image file with the name of the audio file + "_bpm_actual.jpg"
    actual_bpm_filename = os.path.join(data_dir, f"{os.path.splitext(audio_file)[0]}_bpm_actual.jpg")
    plt.savefig(actual_bpm_filename)
    print(f"Live BPM analysis saved as '{actual_bpm_filename}'.")

    plt.close()

def main(url=None):
    if url is None:
        print("No URL provided. Starting microphone recording...")
        duration = int(input("Enter duration for recording (in seconds): "))
        y_recorded_data , sr_recorded_data= record_from_microphone(duration=duration)
        beat_frames = analyze_audio(y_recorded_data , sr_recorded_data )
        bpm_values = calculate_bpm(y_recorded_data , sr_recorded_data , beat_frames)

        mode_value = stats.mode(bpm_values)[0] if len(bpm_values) > 0 else None

        if mode_value is not None:
            print(f'Mode of BPM values: {int(mode_value)}')

        save_actual_bpm_plot(bpm_values , "microphone_recording.wav")
        return

    audio_file = download_youtube_audio(url)

    try:
        y , sr = librosa.load(audio_file)

        plot_fft(y , audio_file)

        beat_frames = analyze_audio(y , sr)

        bpm_values = calculate_bpm(y , sr , beat_frames)

        mode_value = stats.mode(bpm_values)[0] if len(bpm_values) > 0 else None

        if mode_value is not None:
            print(f'Mode of BPM values: {int(mode_value)}')

        plot_waveform_and_bpm(y , sr , bpm_values , audio_file , int(mode_value))

        play_obj = play_audio(audio_file)
        print("Playback started.")

        live_bpm_values = print_bpm_during_playback(librosa.frames_to_time(beat_frames , sr=sr), args.duration)

        play_obj.wait_done()

        save_actual_bpm_plot(live_bpm_values , audio_file)

    finally:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze audio from YouTube or microphone.")

    parser.add_argument('--url', type=str , help="YouTube video URL")
    parser.add_argument('--no-keep', action='store_true', help="Do not keep downloaded files")
    parser.add_argument('--duration', type=int , default=10,
                        help="Duration of real-time analysis in seconds")

    args = parser.parse_args()

    url_input = args.url.strip() if args.url else None

    main(url_input)

def record_from_microphone(duration=10):
    """Record audio from the microphone."""
    import sounddevice as sd

    print("Recording from microphone...")

    # Record audio from microphone
    recording_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                            channels=1, dtype='float64')

    sd.wait()	# Wait until recording is finished

    # Save recorded data as a WAV file (optional)
    wav.write(os.path.join(data_dir,"microphone_recording.wav"), SAMPLE_RATE , recording_data.flatten())

    return recording_data.flatten(), SAMPLE_RATE
