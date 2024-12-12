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

def calculate_bpm_with_windows(y, sr, beat_frames, windows=[1, 2, 5, 10], smoothing_factor=0.1):
    print("Calculating BPM using different window lengths...")
    bpm_values = []

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    for window in windows:
        bpm_array = []
        for i in range(len(beat_times) - 1):
            if beat_times[i + 1] - beat_times[i] <= window:
                bpm_value = (60 / (beat_times[i + 1] - beat_times[i]))  # Calculate BPM between beats

                if 110 <= bpm_value <= 160:
                    bpm_array.append(bpm_value)

        if bpm_array:
            avg_bpm = np.mean(bpm_array)
            bpm_values.append(avg_bpm)

    smoothed_bpm = np.zeros(len(bpm_values))

    smoothed_bpm[0] = bpm_values[0]  # Initialize first value

    for i in range(1, len(bpm_values)):
        smoothed_bpm[i] = (smoothing_factor * bpm_values[i]) + ((1 - smoothing_factor) * smoothed_bpm[i - 1])

    final_bpm = round(smoothed_bpm[-1])  # Return rounded final value
    print(f"Final estimated BPM after smoothing: {final_bpm}")

    return final_bpm

def format_time(seconds):
    """Convert seconds into a string formatted as MM:SS."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}"

def plot_waveform_and_beats(y, sr, beat_frames, audio_file):
    print("Plotting waveform and beats...")

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')

    if beat_frames.size > 0:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        plt.subplot(2, 1, 2)
        plt.vlines(beat_times, 0, 1, color='r', alpha=0.5, label='Beats')
        plt.title('Beat Times')
        plt.xlabel('Time (s)')
        plt.ylabel('Beats')
        plt.legend()

    plt.tight_layout()

    # Save plots as an image file with the name of the audio file + "_waveform.jpg"
    plot_filename = f"{os.path.splitext(audio_file)[0]}_waveform.jpg"
    plt.savefig(plot_filename, format='jpg')
    print(f"Plots saved as '{plot_filename}'.")

    plt.close()

def play_audio(file_path):
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    return play_obj

def print_bpm_during_playback(beat_times, duration):
    start_time = time.time()
    print("Calculating BPM during playback...")

    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time

        window_start_time = elapsed_time - 5.0

        recent_beats = [bt for bt in beat_times if window_start_time < bt <= elapsed_time]

        if len(recent_beats) > 1:
            bpm_count = len(recent_beats) - 1
            time_interval = recent_beats[-1] - recent_beats[0]

            if time_interval > 0:
                current_bpm = round((bpm_count / time_interval) * 60.0)
                print(f'Current BPM at {int(elapsed_time)}s: {current_bpm} BPM')  # Truncate time and round BPM

        time.sleep(1)

if __name__ == "__main__":
    url = input("Enter the YouTube video URL: ")

    keep_original = input("Do you want to keep original files? (y/n): ").strip().lower() == 'y'

    audio_file = download_youtube_audio(url, keep_original)

    try:
        y, sr = librosa.load(audio_file)
        beat_frames = analyze_audio(y, sr)

        plot_waveform_and_beats(y,sr,beat_frames,audio_file)

        playback_duration = librosa.get_duration(y=y,sr=sr)

        play_obj = play_audio(audio_file)

        bpm_thread = threading.Thread(target=print_bpm_during_playback,
                                      args=(librosa.frames_to_time(beat_frames,sr=sr), playback_duration))
        bpm_thread.start()

        play_obj.wait_done()
        bpm_thread.join()

    finally:
        pass

