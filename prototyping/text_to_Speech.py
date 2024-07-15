import torch 
import pyaudio
import wave
from scipy.io.wavfile import write
import re
import numpy as np
from TTS.api import TTS
import bark
import os
from transformers import BarkModel,AutoProcessor
import nltk
import monsterapi
import requests


RESPONSE_TEXT = "Nice to meet you, Connie! I'm doing great, thanks for asking! As a conversational AI assistant, my purpose is to assist and provide helpful information, so it's always exciting to start a new conversation with someone like you. I'd love to get to know more about what brings you here today. Is there something specific on your mind that you'd like to talk about or ask me? Or perhaps we can explore some topics together? By the way, I hope you don't mind my saying so, but it's lovely to have a chat with someone named Connie - it's not every day I get to meet someone with such a unique and charming name!"
def split_sentence(text):
    regex = '(?<=[\.\?\!])\s*'
    return re.split(regex, text)

def initialize_tacotron():
    tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math = 'fp16')
    tacotron2.eval()
    return tacotron2

def initialize_waveglow():
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()
    return waveglow

def run_inference (text, taco, waveglow, filename) :
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
    sequences, lengths = utils.prepare_input_sequence([text],cpu_run = True)
    with torch.no_grad():
        mel, _, _ = taco.infer(sequences, lengths)
        audio = waveglow.infer(mel)
    
    audio_numpy = audio[0].data.cpu().numpy()
    write(filename, rate, audio_numpy)

    '''
    with wave.open('reording.wav','wb') as wf:
        pyAudio = pyaudio.PyAudio()
        wf.setnchannels(1)
        wf.setframerate(rate)
        wf.setsampwidth(pyAudio.get_sample_size(pyaudio.paFloat32))
        wf.writeframes(audio_numpy.tobytes())
    '''
    return audio_numpy


def play_audio(audio_content, rate):

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=rate,
                         output=True)
    stream.write(audio_content.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

    '''
    with wave.open(audio_content, 'rb') as wf:
        p = pyaudio.PyAudio()
        print(wf.getsampwidth()) #2
        print(wf.getnchannels()) #1
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=22050,
            output=True
        )
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
    '''


if __name__ =="__main__":
    #taco = initialize_tacotron()
    #wg = initialize_waveglow()

    #api_key = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImMwMDE2Y2I1YTU2MzZkMmY5YjgxNDE5ZGJlMWEwMDYxIiwiY3JlYXRlZF9hdCI6IjIwMjQtMDctMTRUMTc6MjI6MjMuMDE0Njg1In0.6Xt8VjovN8fm7MIF1R2_gJkzW0zYQ5ZXs8TIn_RSAZM'
    #client = monsterapi.client(api_key=api_key)

    #model='sunoai-bark'

    #script = RESPONSE_TEXT.replace("\n", " ").strip()
    #sentences = nltk.sent_tokenize(script)
    #print(sentences)

    '''
    recording = []
    for i, phrase in enumerate(sentences[:-1]):
        recording.append(run_inference(phrase, taco, wg, f'audio_{i}.wav'))
        
    play_audio(np.concatenate(recording),rate)
    '''
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #os.environ["SUNO_USE_SMALL_MODELS"] = "1"
    #device = 'cpu'
    #TTS().list_models()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    #processor = AutoProcessor.from_pretrained("suno/bark", force_download = True)
    #model = BarkModel.from_pretrained("suno/bark", force_download=True)
    #voice_preset = "en_speaker_2"
    #inputs = processor(RESPONSE_TEXT,voice_preset = voice_preset)

    '''
    input_data = {
            'prompt': RESPONSE_TEXT,
            'sample_rate': bark.SAMPLE_RATE,
            'speaker': voice_preset,
            'text_temp': 0.5,
            'wave_temp': 0.5
        }
    result = client.generate(model,input_data)
    audio_file = requests.get(result['content']).content
    play_audio(audio_file, bark.SAMPLE_RATE)
    '''
 
    #bark.preload_models()

    audio_file = tts.tts(text=RESPONSE_TEXT,
        speaker='Ana Florence',
        language="en",
        split_sentences=True
        )
    

    #write(f"output.wav", rate=bark.SAMPLE_RATE, data=np.array(audio_file, dtype=np.float32))
    play_audio(np.array(audio_file, dtype=np.float32),bark.SAMPLE_RATE)
    


    '''
    for i, phrase in enumerate(sentences):
        #audio_array = bark.generate_audio(phrase, text_temp=0.5, waveform_temp=0.5, history_prompt=voice_preset)
        #result = client.generate(model,input_data)
        #print(result['output'])

        #write(f"bark_out_{i}.wav", rate=bark.SAMPLE_RATE, data=result['output'])
        #play_audio(result['output'],bark.SAMPLE_RATE)
'''
    