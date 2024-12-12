import yt_dlp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import simpleaudio as sa
import time
import argparse

def download_youtube_audio(url, keep_original=True):
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
    audio_file_wav = f"{title}.wav"
    
    # Check if the WAV file already exists
    if os.path.exists(audio_file_wav):
        print(f"File '{audio_file_wav}' already exists. Skipping download.")
        return audio_file_wav  # Return existing WAV file name

    # Download audio if it does not exist
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return audio_file_wav  # Return the filename of the downloaded audio

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

    return beat_frames, tempo

def calculate_bpm_at_samples(y, sr, beat_frames, smoothing_factor=0.1):
    # Create a time array for the audio signal
    duration = len(y) / sr
    time_array = np.linspace(0, duration, len(y))

    # Create an empty BPM array initialized to zero
    bpm_array = np.zeros_like(time_array)

    # Calculate BPM values based on detected beats
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    for i in range(len(beat_times) - 1):
        bpm_value = (60 / (beat_times[i + 1] - beat_times[i]))  # Calculate BPM between beats
        
        # Fill the bpm_array with this value for the duration between beats
        bpm_array[(time_array >= beat_times[i]) & (time_array < beat_times[i + 1])] = bpm_value

    # Apply exponential smoothing to stabilize BPM values
    smoothed_bpm_array = np.zeros_like(bpm_array)
    
    for i in range(len(bpm_array)):
        if i == 0:
            smoothed_bpm_array[i] = bpm_array[i]
        else:
            smoothed_bpm_array[i] = (smoothing_factor * bpm_array[i]) + ((1 - smoothing_factor) * smoothed_bpm_array[i - 1])

    return smoothed_bpm_array

def plot_waveform_and_bpm(y, sr, bpm_array, audio_file):
    # Plotting waveform and BPM over time
    plt.figure(figsize=(12, 6))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    
    # Plot BPM(t)
    plt.subplot(2, 1, 2)
    time_array = np.linspace(0, len(y) / sr, len(y))
    
    plt.plot(time_array, bpm_array, color='r', label='BPM(t)')
    plt.title('BPM over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('BPM')
    
    # Set y-axis limits to zoom in on values between 110 and 160 BPM with a clear separation of 5 BPM
    plt.ylim(110, 160)
    
    # Set y-ticks for better readability (from 110 to 160 with a step of 5)
    plt.yticks(np.arange(110, 165, step=5))
    
    plt.legend()
    
    plt.tight_layout()
    
    # Save plots as an image file with the name of the audio file + "_waveform.jpg"
    plot_filename = f"{os.path.splitext(audio_file)[0]}_waveform.jpg"
    plt.savefig(plot_filename, format='jpg')  # Save the figure as a JPEG file
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze audio from YouTube or microphone.")
    
    parser.add_argument('--url', type=str, help="YouTube video URL")
    parser.add_argument('--no-keep', action='store_true', help="Do not keep downloaded files")
    
    args = parser.parse_args()

    if args.url:
        url = args.url.strip()
        
        # Download audio from YouTube video or use existing WAV/MP3 file
        audio_file = download_youtube_audio(url)

        try:
            y, sr = librosa.load(audio_file)
            beat_frames, tempo = analyze_audio(y, sr)

            bpm_array = calculate_bpm_at_samples(y, sr, beat_frames)

            plot_waveform_and_bpm(y, sr, bpm_array, audio_file)  # Save plot before playback

            playback_duration = librosa.get_duration(y=y, sr=sr)  
            play_obj = play_audio(audio_file)
            
            print_bpm_during_playback(librosa.frames_to_time(beat_frames, sr=sr), playback_duration)

            play_obj.wait_done()  
            
        finally:
            if not args.no_keep and os.path.exists(audio_file):
                print(f"Keeping file: {audio_file}")
            else:
                os.remove(audio_file)  # Remove the file if --no-keep is specified
                
    else:
        url_input = input("No URL provided. Enter a YouTube video URL or press Enter to record from microphone: ").strip()
        
        while not url_input:  # Keep asking until a valid URL is provided or recording is chosen.
            print("No URL provided.")
            url_input = input("Enter a YouTube video URL or press Enter to record from microphone: ").strip()

        if url_input: 
            audio_file = download_youtube_audio(url_input)

            try:
                y, sr = librosa.load(audio_file)
                beat_frames, tempo = analyze_audio(y, sr)

                bpm_array = calculate_bpm_at_samples(y, sr, beat_frames)

                plot_waveform_and_bpm(y, sr, bpm_array, audio_file)  # Save plot before playback

                playback_duration = librosa.get_duration(y=y, sr=sr)  
                play_obj = play_audio(audio_file)

                print_bpm_during_playback(librosa.frames_to_time(beat_frames, sr=sr), playback_duration)

                play_obj.wait_done()  

            finally:
                if not args.no_keep and os.path.exists(audio_file):
                    print(f"Keeping file: {audio_file}")
                else:
                    os.remove(audio_file)  # Remove the file if --no-keep is specified

        else:
            y, sr = record_audio(duration=10)  # Record for 10 seconds
            beat_frames = analyze_audio(y, sr)

            plot_waveform_and_beats(y, sr, beat_frames, "recorded_audio.wav")  # Save plot for recorded audio

            playback_duration = len(y) / sr  
            
            sd.play(y, sr)
            
            print_bpm_during_playback(librosa.frames_to_time(beat_frames, sr=sr), playback_duration)

