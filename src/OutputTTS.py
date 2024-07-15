import pyaudio
import re
import numpy as np
from TTS.api import TTS

class outputTTS():
    SAMPLE_RATE = 24000

    def __init__(self, rate = SAMPLE_RATE) -> None:
        self.rate = rate
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        self.llmresponse = ""

    def run_inference (self, response:str, sentences = True): 
        self.llmresponse = response
        result = self.model.tts(
            text=response,
            speaker='Ana Florence',
            language="en",
            split_sentences=sentences
        )

        result = np.array(result, dtype=np.float32)
        return result
    
    def play_audio(self, audio):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.rate,
                        output=True)
        stream.write(audio.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def split_into_sentence(text):
        regex = '(?<=[\.\?\!])\s*'
        return re.split(regex, text)[:-1]
    
    

