import yt_dlp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import simpleaudio as sa
import time
import argparse
import sounddevice as sd

def download_youtube_audio(url):
    # Set options for yt-dlp to download audio only
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Save as WAV for better compatibility with librosa
            'preferredquality': '192',
        }],
        'outtmpl': '%(title)s.%(ext)s',  # Output filename template using video title
    }
    
    # Extract video info to get the title before downloading
    info_dict = yt_dlp.YoutubeDL().extract_info(url, download=False)
    title = info_dict.get('title', None)
    audio_file = f"{title}.wav"

    # Check if the file already exists
    if os.path.exists(audio_file):
        print(f"File '{audio_file}' already exists. Skipping download.")
        return audio_file  # Return existing file name

    # Download audio if it does not exist
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return audio_file  # Return the filename of the downloaded audio

def analyze_audio(y, sr):
    # Calculate BPM using librosa's beat detection
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    # Check if tempo is valid and print estimated BPM
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]  # Get the first element if it's an array

    if tempo > 0:
        print(f'Estimated BPM: {tempo:.2f}')
    else:
        print('No beats detected or unable to estimate BPM.')

    return beat_frames

def plot_waveform_and_beats(y, sr, beat_frames, audio_file):
    # Plotting waveform and BPM over time
    plt.figure(figsize=(12, 6))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    
    # Plot BPM over time if beats were detected
    if beat_frames.size > 0:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        plt.subplot(2, 1, 2)
        plt.vlines(beat_times, 0, 1, color='r', alpha=0.5, label='Beats')
        plt.title('Beat Times')
        plt.xlabel('Time (s)')
        plt.ylabel('Beats')
        plt.legend()
    
    plt.tight_layout()
    
    # Save plots as an image file with the name of the audio file + "_waveform"
    plot_filename = f"{os.path.splitext(audio_file)[0]}_waveform.png"
    plt.savefig(plot_filename)  # Save the figure as a PNG file
    print(f"Plots saved as '{plot_filename}'.")
    
    plt.close()  # Close the plot to free up memory

def play_audio(file_path):
    # Load and play audio using simpleaudio
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    
    return play_obj

def print_bpm_during_playback(beat_times, duration):
    start_time = time.time()
    
    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        
        window_start_time = elapsed_time - 5.0
        
        recent_beats = [bt for bt in beat_times if window_start_time < bt <= elapsed_time]
        
        if len(recent_beats) > 1:  # Need at least two beats to calculate BPM
            bpm_count = len(recent_beats) - 1  
            time_interval = recent_beats[-1] - recent_beats[0]  
            
            if time_interval > 0:
                current_bpm = (bpm_count / time_interval) * 60.0  
                print(f'Current BPM at {elapsed_time:.2f}s: {current_bpm:.2f} BPM')
        
        time.sleep(0.5)

def record_audio(duration=10):
    print("Recording...")
    fs = 44100  # Sample rate
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    
    return myrecording.flatten(), fs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze audio from YouTube or microphone.")
    
    parser.add_argument('-k', '--youtube', type=str, help="YouTube video URL")
    
    args = parser.parse_args()

    if args.youtube is not None:
        url = args.youtube.strip() or None
        
        if not url:  # If an empty URL is passed with -k option
            print("No URL provided. Starting to listen through microphone...")
            y, sr = record_audio(duration=10)  # Record for 10 seconds
            beat_frames = analyze_audio(y, sr)

            plot_waveform_and_beats(y, sr, beat_frames, "recorded_audio.wav")  # Save plot for recorded audio

            playback_duration = len(y) / sr  
            
            sd.play(y, sr)
            
            print_bpm_during_playback(librosa.frames_to_time(beat_frames, sr=sr), playback_duration)

        else:
            # Download audio from YouTube video or use existing file
            audio_file = download_youtube_audio(url)
            
            try:
                y, sr = librosa.load(audio_file)
                beat_frames = analyze_audio(y, sr)

                plot_waveform_and_beats(y, sr, beat_frames, audio_file)

                playback_duration = librosa.get_duration(y=y, sr=sr)  
                play_obj = play_audio(audio_file)
                
                print_bpm_during_playback(librosa.frames_to_time(beat_frames, sr=sr), playback_duration)

                play_obj.wait_done()  
                
            finally:
                if os.path.exists(audio_file):
                    os.remove(audio_file)

    else:
        url_input = input("Enter the YouTube video URL: ").strip()
        
        while not url_input:  # Keep asking until a valid URL is provided
            url_input = input("You must enter a URL. Please enter the YouTube video URL: ").strip()

        audio_file = download_youtube_audio(url_input)
        
        try:
            y, sr = librosa.load(audio_file)
            beat_frames = analyze_audio(y, sr)

            plot_waveform_and_beats(y, sr, beat_frames, audio_file)

            playback_duration = librosa.get_duration(y=y, sr=sr)  
            play_obj = play_audio(audio_file)
            
            print_bpm_during_playback(librosa.frames_to_time(beat_frames, sr=sr), playback_duration)

            play_obj.wait_done()  
            
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)

