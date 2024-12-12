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

def download_youtube_audio(url, keep_original=False):
    print("Starting download...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Save as WAV for better compatibility with librosa
            'preferredquality': '192',
        }],
        'outtmpl': '%(title)s.%(ext)s',  # Output filename template using video title
    }

    info_dict = yt_dlp.YoutubeDL().extract_info(url, download=False)
    title = info_dict.get('title', None)
    audio_file_wav = f"{title}.wav"
    audio_file_mp3 = f"{title}.mp3"

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

        if bpm_value > 0:  # Only consider positive BPM values
            if not bpm_values or bpm_value != bpm_values[-1]:  # Store only if it changes
                bpm_values.append(bpm_value)

    # Smooth the BPM values using a simple moving average or exponential smoothing
    if len(bpm_values) > 0:
        smoothed_bpm = np.convolve(bpm_values, np.ones(5)/5, mode='valid')  # Simple moving average
        return smoothed_bpm

    return np.array([])  # Return an empty array if no BPM values were calculated

def plot_waveform_and_bpm(y, sr, bpm_values, audio_file):
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

    plt.ylim(0, max(bpm_values) + 10)  # Set y-limits based on max BPM value

    plt.legend()

    # Save plots as an image file with the name of the audio file + "_waveform_bpm.jpg"
    plot_filename = f"{os.path.splitext(audio_file)[0]}_waveform_bpm.jpg"
    plt.savefig(plot_filename, format='jpg')
    print(f"Plots saved as '{plot_filename}'.")

    plt.close()

def plot_fft(bpm_values):
    """Plot FFT of BPM values."""
    N = len(bpm_values)
    T = 1.0 / (60.0 / np.mean(bpm_values))  # Sampling interval based on average BPM
    yf = np.fft.fft(bpm_values)
    xf = np.fft.fftfreq(N, T)[:N // 2]

    plt.figure(figsize=(12, 6))
    plt.plot(xf[(xf >= (110/60)) & (xf <= (160/60))],
             np.abs(yf[:N // 2][(xf >= (110/60)) & (xf <= (160/60))]), color='b')
    plt.title('Fourier Transform of BPM Values (110-160 BPM)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.xlim(110/60, 160/60)
    plt.xticks(np.arange(110/60, 160/60 + (1/60), step=1/60))

    fft_filename = "file_fft.jpg"
    plt.savefig(fft_filename)
    print(f"FFT plot saved as '{fft_filename}'.")

    plt.close()

def play_audio(file_path):
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    return play_obj

def print_bpm_during_playback(beat_times, duration):
    start_time = time.time()
    last_printed_bpm = None  # Variable to track the last printed BPM value

    print("Calculating BPM during playback...")

    live_bpm_values = []  # List to store real-time BPM values

    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time

        window_start_time = elapsed_time - 5.0

        recent_beats = [bt for bt in beat_times if window_start_time < bt <= elapsed_time]

        if len(recent_beats) > 1:
            bpm_count = len(recent_beats) - 1
            time_interval = recent_beats[-1] - recent_beats[0]

            if time_interval > 0:
                current_bpm = round((bpm_count / time_interval) * 60.0)
                if current_bpm != last_printed_bpm:  # Only print when it changes
                    print(f'Current BPM at {int(elapsed_time)}s: {current_bpm} BPM')
                    last_printed_bpm = current_bpm
                    live_bpm_values.append(current_bpm)

        time.sleep(1)  # Sleep for one second

    return live_bpm_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze audio from YouTube or microphone.")

    parser.add_argument('--url', type=str, required=True, help="YouTube video URL")
    parser.add_argument('--no-keep', action='store_true', help="Do not keep downloaded files")

    args = parser.parse_args()

    url = args.url.strip()

    audio_file = download_youtube_audio(url, not args.no_keep)

    try:
        y, sr = librosa.load(audio_file)
        beat_frames = analyze_audio(y, sr)

        bpm_values = calculate_bpm(y,sr,beat_frames)

        plot_waveform_and_bpm(y,sr,bpm_values,audio_file)

        playback_duration = librosa.get_duration(y=y,sr=sr)

        live_bpm_values = print_bpm_during_playback(librosa.frames_to_time(beat_frames,sr=sr), playback_duration)

        plot_fft(live_bpm_values)

        play_obj = play_audio(audio_file)

        bpm_thread = threading.Thread(target=print_bpm_during_playback,
                                      args=(librosa.frames_to_time(beat_frames,sr=sr), playback_duration))
        bpm_thread.start()

        play_obj.wait_done()
        bpm_thread.join()

    finally:
        pass
