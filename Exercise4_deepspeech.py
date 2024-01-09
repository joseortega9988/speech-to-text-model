# Import necessary libraries for audio processing and DeepSpeech
import scipy
import wave
import math
import librosa
import numpy as np
from scipy.io import wavfile
import scipy.signal

# Please ensure you are using a Python environment between versions
# 3.5 and 3.9, as the deepspeech library is not compatible 
#with Python 3.10 or later versions.

import deepspeech

# command to install all the libraries for this script 
# pip install scipy wave librosa numpy deepspeech


# Display a menu for language selection
print("Seleccione su idioma **************** Select your language **************** Seleziona la tua lingua")
language_valid = False
language = ""

# Loop to ensure valid language input
while not language_valid:
    user_language = input("Español(ES), English(EN), Italiana(it) : ")
    try:
        # Normalize user input to lower case for easier comparison
        user_language = user_language.lower()

        # Check for different ways to write the language and set up the corresponding language model
        if user_language in ["en", "eng", "english", "ingles", "inglese"]:
            language = "ENG"
            path_of_the_model = 'language_models/deepspeech-0.9.3-models.pbmm'
            language_models = deepspeech.Model(path_of_the_model)
            path_scorer = 'language_models/deepspeech-0.9.3-models.scorer'
            language_valid = True

        elif user_language in ["es", "esp", "español", "spanish", "espanol", "spagnola"]:
            language = "ESP"
            path_of_the_model = 'language_models/output_graph_es.pbmm'
            language_models = deepspeech.Model(path_of_the_model)
            path_scorer = 'language_models/kenlm_es.scorer'
            language_valid = True

        elif user_language in ["it", "ita", "italiana", "italiano", "italian"]:
            language = "ITA"
            path_of_the_model = 'language_models/output_graph_it.pbmm'
            language_models = deepspeech.Model(path_of_the_model)
            path_scorer = 'language_models/kenlm_it.scorer'
            language_valid = True

        # Prompt for correct input if invalid language is entered
        if not language_valid:
            print("Please enter a valid language **************** Por Favor Introduzca un idioma válido **************** Per favore una lingua valida ")

    except Exception as e:
        print("Error related to the model; please check the model location and all the model archives.")

# Configure the DeepSpeech model with the external scorer for better accuracy
language_models.enableExternalScorer(path_scorer)
lang_model_beta = 1.85
lang_model_alpha = 0.75
language_models.setScorerAlphaBeta(lang_model_alpha, lang_model_beta)
width_of_beam = 500
language_models.setBeamWidth(width_of_beam)
ideal_sample_rate = language_models.sampleRate()


# Define paths to audio files based on selected language

if language == "ESP": # Add paths for Spanish audio files
    audiofiles = [
                'Ex4_audio_files/ES/parents_es.wav', #1
                'Ex4_audio_files/ES/suitcase_es.wav', #2
                'Ex4_audio_files/ES/checkin_es.wav', #3
                'Ex4_audio_files/ES/where_es.wav',#4
                'Ex4_audio_files/ES/what_time_es.wav', #5

                ] 
        
if language == "ITA": # Add paths for Italian audio files
    audiofiles = [
                'Ex4_audio_files/IT/parents_it.wav', #1
                'Ex4_audio_files/IT/suitcase_it.wav', #2
                'Ex4_audio_files/IT/checkin_it.wav', #3
                'Ex4_audio_files/IT/where_it.wav', #4
                'Ex4_audio_files/IT/what_time_it.wav' #5

                ] 
    
if language == "ENG": # Add paths for English audio files
    audiofiles = [
                'Ex4_audio_files/EN/parents.wav', #1
                'Ex4_audio_files/EN/suitcase.wav', #2
                'Ex4_audio_files/EN/checkin.wav', #3
                'Ex4_audio_files/EN/where.wav', #4
                'Ex4_audio_files/EN/what_time.wav', #5


                 ]
converted_texts = []

# Function to apply a bandpass filter to audio data
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    low = lowcut / nyquist  # Normalize low cutoff frequency
    high = highcut / nyquist  # Normalize high cutoff frequency
    b, a = scipy.signal.butter(order, [low, high], btype='band')  # Create bandpass filter
    y = scipy.signal.lfilter(b, a, data)  # Apply filter to data
    return y
# content download from https://www.youtube.com/watch?v=2dWlKLCKW_Q
# Load a crowd ambience audio file to use for noise reduction
g = 1
crowd_ambience_audio = 'Ex4_audio_files/EN/crowd_ambience.wav'
ats, y_samping_rate = librosa.load(crowd_ambience_audio, sr=ideal_sample_rate)
# Perform Short-Time Fourier Transform (STFT) on the ambience audio
STFT_audio = librosa.stft(ats)
magnitud_spec = np.abs(STFT_audio)
avg_mag_spec_over_time = np.mean(magnitud_spec, axis=1)

# Iterate through each audio file for the selected language
for filename in audiofiles:
    # Load the audio file

    file_ats, y_sr = librosa.load(filename, sr=None, mono=True)
    # Perform STFT on the audio file
    file_ats_alterated = bandpass_filter(file_ats, 200, 3000, 44100, 1)
    STFT_audio = librosa.stft(file_ats)
    STFT_transformed = np.abs(STFT_audio)  # Get the magnitude spectrum
    angle = np.angle(STFT_audio)  # Get the phase information
    k = np.exp(1.0j * angle)  # Convert phase information to complex numbers



    # Noise Reduction: Subtract crowd noise from the audio
    audio_substracted = STFT_transformed - avg_mag_spec_over_time.reshape((avg_mag_spec_over_time.shape[0], 1))
    audio_substracted = audio_substracted * k  # Combine with original phase information
    a = librosa.istft(audio_substracted)  # Inverse STFT to convert back to audio time series


    # Write the processed audio to a new file
    scipy.io.wavfile.write("Ex4_audio_files/mywav_reduced_noise" + str(g) + ".wav", y_sr, (a * 32768).astype(np.int16))

    # perform low pass filter on reduced crowd noise file
    freq_sampling_rate, data = wavfile.read("Ex4_audio_files/mywav_reduced_noise" + str(g) + ".wav")

    # different cutoff frequency for different languages, explanation in report
    if  language == "ENG"  or language == "ITA":
        cut_off_frequency = 3000.0
    if language == "ESP"  :
        cut_off_frequency = 3000.0

    ratio_frequency = cut_off_frequency / freq_sampling_rate

    size_of_filter_window = int(math.sqrt(0.196201 + ratio_frequency**2) / ratio_frequency)

    filter_window = np.ones(size_of_filter_window)
    filter_window *= 1.0/size_of_filter_window
    # low pass filter
    low_p_filter = scipy.signal.lfilter(filter_window, [1], data).astype(np.int16)

    # Retrieve properties from the audio file with reduced noise
    wave_open = wave.open(f"Ex4_audio_files/mywav_reduced_noise{g}.wav", 'r')
    amplitude_width = wave_open.getsampwidth()  # Get the sample width (in bytes) of the audio
    number_frames = wave_open.getnframes()  # Get the total number of frames in the audio
    wave_open.close()  # Close the file after retrieving information

    # Write a new audio file after applying a low-pass filter
    wav_audio = wave.open(f"Ex4_audio_files/mywav_reduced_noise{g}filtered.wav", "w")
    wav_audio.setnchannels(1)  # Set the number of channels (mono in this case)
    wav_audio.setsampwidth(amplitude_width)  # Set the sample width
    wav_audio.setframerate(ideal_sample_rate)  # Set the frame rate (sampling rate)
    wav_audio.setnframes(number_frames)  # Set the total number of frames
    wav_audio.writeframes(low_p_filter.tobytes('C'))  # Write the filtered audio data to the file
    wav_audio.close()  # Close the file

    # Open the newly written filtered audio file
    wave_open = wave.open(f"Ex4_audio_files/mywav_reduced_noise{g}filtered.wav", 'r')
    frames_number = wave_open.getnframes()  # Get the number of frames in the audio
    frames = wave_open.readframes(frames_number)  # Read all frames of the audio

    # Convert the audio frames to 16-bit data and perform speech-to-text using DeepSpeech
    bit16_data = np.frombuffer(frames, dtype=np.int16)
    text_from_speech = language_models.stt(bit16_data)
    converted_texts.append(text_from_speech)  # Append the recognized text to the list
    wave_open.close()  # Close the audio file
    g += 1  # Increment file counter for output file naming


# Check the selected language and set up the corresponding transcriptions for each audio file
if language == "ENG":
    # English transcriptions
    transcription_of_text = [
                        "I have lost my parents.", #1
                        "Please, I have lost my suitcase.", #2
                        "Where is the check-in desk?", #3
                        "Where are the restaurants and shops?", #4
                        "What time is my plane?", #5
                        ]
if language == "ESP":  
    # Spanish transcriptions
    transcription_of_text = [
                        "He perdido a mis padres.", #1dd
                        "Por favor, he perdido mi maleta.", #2dd
                        "¿Dónde están los mostradores?", #3 ddd
                        "¿Dónde están los restaurantes y las tiendas?", #4
                        "¿A qué hora es mi avión?" #5

                        ] 
if language == "ITA":
    # Italian transcriptions
    transcription_of_text = [
                        "Ho perso i miei genitori.", #1
                        "Per favore, ho perso la mia valigia.", #2 
                        "Dove e' il bancone?", #3
                        "Dove sono i ristoranti e i negozi?", #4
                        "A che ora e' il mio aereo?"#5
                        ]
    
# Iterate through each predefined transcription

for i, text in enumerate(transcription_of_text):
    # Remove punctuation marks and convert the text to lower case
    # This normalization step ensures consistency for comparison
    converter_text = "".join(character for character in text if character not in ("?", ".", ";", ":", "!", ",", "¿", "'"))
    
    # Replace hyphens with spaces to standardize the text format
    converter_text = converter_text.replace("-", " ")
    
    # Convert the text to lower case to avoid case sensitivity issues during comparison
    converter_text = converter_text.lower()
    
    # Update the transcription list with the normalized text
    transcription_of_text[i] = converter_text

# Initialize variables for total errors and total words
sum_of_sub_and_del = 0
N = 0
total_WER = 0
total_SDI = 0
total_N = 0

# Iterate through each recognized text and its corresponding transcription
for i, text in enumerate(converted_texts):
    print("Recognised text: " + text)
    print("Transcript text: " + transcription_of_text[i])

    # Split the recognized and transcribed texts into words
    list_of_text = text.split(" ")
    list_of_transcript_text = transcription_of_text[i].split(" ")
    
    # Calculate substitution and deletion errors
    # Substitution errors occur when words in the transcription don't match the recognized text
    # Deletion errors occur when words in the transcription are missing in the recognized text
    substitution_and_deletion_errors = len(list_of_transcript_text) - len(set(list_of_text).intersection(list_of_transcript_text))
    
    # Calculate insertion errors
    # Insertion errors occur when the recognized text contains extra words not in the transcription
    difference_length = len(list_of_transcript_text) - len(list_of_text)
    insertion_errors = abs(difference_length) if difference_length < 0 else 0
    
    # Update the total errors and word count
    sum_of_sub_and_del = substitution_and_deletion_errors + insertion_errors
    N = len(list_of_transcript_text)
    print("Number of errors: " + str(sum_of_sub_and_del))
    print("Total number of words: " + str(N))
    
    # Calculate the WER for the current transcript and update totals
    actual_WER = sum_of_sub_and_del / N * 100
    print("WER for current transcript: " + str(actual_WER) + "%")
    total_SDI += sum_of_sub_and_del
    total_N += N

# Calculate the overall WER across all transcripts
total_WER = total_SDI / total_N * 100
print("Overall " + language + " WER: " + str(total_WER) + "%")