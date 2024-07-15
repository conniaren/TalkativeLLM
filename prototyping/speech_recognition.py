import os 
import azure.cognitiveservices.speech as speechsdk
import wave 
import sys
import pyaudio
from deepspeech import Model
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 16000
RECORD_SECS = 30
AUDIO_FILE_PATH = './audio/files/record_output.wav'
MODEL_FILE_PATH = './models/deepspeech-0.9.3-models.pbmm'
LANGUAGE_FILE_PATH = './models/deepspeech-0.9.3-models.scorer'
BEAM_WIDTH = 100
ALPHA = 0.93
BETA = 1.18


def initialize_DS_model():
    model = Model(MODEL_FILE_PATH)

    model.enableExternalScorer(LANGUAGE_FILE_PATH)
    model.setBeamWidth(BEAM_WIDTH)
    model.setScorerAlphaBeta(ALPHA,BETA)

    return model


def record_audio():
    with wave.open(AUDIO_FILE_PATH,'wb') as wf:
        pyAudio = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setframerate(RATE)
        wf.setsampwidth(pyAudio.get_sample_size(FORMAT))

        stream = pyAudio.open(format=FORMAT, channels = CHANNELS, rate = RATE, input = True) #streaming service

        print(" Recording ... ")
        for _ in range(0, RATE//CHUNK*RECORD_SECS):
            wf.writeframes(stream.read(CHUNK))

        print(" Finish Recording. ")
        stream.stop_stream()
        stream.close()
        pyAudio.terminate()
        wf.close()


def stream_audio():
    stream_model = initialize_DS_model()
    

    with wave.open(AUDIO_FILE_PATH, 'rb') as wf:
        buffer = wf.readframes(wf.getnframes()) 
        wf.close()          
        
    start = 0 
    print(" Streaming Speech-to-Text ... ")
    while start<len(buffer):
        end = start+CHUNK
        chunk = buffer[start:end]
        data16 = np.frombuffer(chunk, dtype=np.int16)
        #print(f"Feeding chunk from {start} to {end}, data16 length: {len(data16)}, dtype: {data16.dtype}")

        try:
            result = stream_model.stt(data16)
            print(result)
        except Exception as e:
            print(f"Error feeding audio content: {e}")
            break
        start = end



        
    
   #try:
        #final_result = stream_model.finishStream()
       # print(final_result)
    #except Exception as e:
        #print(f"Error finishing stream: {e}")
    print(" Finish Stream. ")
        



def recognize_from_file():
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get("SPEECH_KEY"),
                                           region=os.environ.get("SPEECH_REGION"))
    speech_config.speech_recognition_language = "en-GB"
    audio_config = speechsdk.audio.AudioConfig(filename=AUDIO_FILE_PATH)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                                           audio_config=audio_config)
    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    process_Speech(speech_recognition_result)





def recognize_from_mic():
    print(os.environ.get("SPEECH_KEY"))
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get("SPEECH_KEY"),
                                           region=os.environ.get("SPEECH_REGION"))
    speech_config.speech_recognition_language = "en-GB"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                                           audio_config=audio_config)
    
    print("Speak into your microphone! ")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    process_Speech(speech_recognition_result)


def process_Speech(result):
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

if __name__ == "__main__":
    #record_audio()
    stream_audio()
    #recognize_from_mic()
    #recognize_from_file()
    #transcribe_streaming(AUDIO_FILE_PATH)
