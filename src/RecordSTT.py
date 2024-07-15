from threading import Thread
from queue import Queue
import pyaudio
import time
import numpy as np
import whisper
import keyboard


class recordSTT():

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECS = 5
    SAMPLE_SIZE = 2

    def __init__(self) -> None:
        self.model = whisper.load_model('base.en')
        self.messages = Queue()
        self.recordings = Queue()
        self.final_string = []
        self.final_output_string = []
        self.is_recording = False

    def record_mic(self, chunk = CHUNK, format = FORMAT, channels = CHANNELS, rate = RATE, secs = RECORD_SECS):
        p = pyaudio.PyAudio()
        stream = p.open(format=format, 
                        channels = channels, 
                        rate = rate, 
                        input = True, 
                        frames_per_buffer=chunk)

        frames = []
        while self.is_recording:
            data = stream.read(chunk)
            frames.append(data)
    
            if len(frames) >= rate//chunk*secs:
                self.recordings.put(frames.copy())
                frames = []
            
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def speech_recognition(self):
        while not self.recordings.empty() or self.is_recording:
            if not self.recordings.empty():
                frames = self.recordings.get()

                data16 = np.frombuffer(b''.join(frames), dtype=np.int16)
                data32 = data16.astype(np.float32)/32768.0
                
                result = self.model.transcribe(data32, fp16=False)
                self.final_output_string.append(result["text"]+'\n')
                self.final_string.append(result['text'])


                print(result["text"]+'\n')
                time.sleep(1)
    
    def start_recording(self):
        self.is_recording = True

        print("Start speaking... Press Enter to stop. \n")
        record = Thread(target = self.record_mic)
        record.start()

        transcribe = Thread(target = self.speech_recognition)
        transcribe.start()
        while True:
            if keyboard.is_pressed('enter'):
                self.is_recording = False
                print(" Voice Captured. \n")
                break

        record.join()
        transcribe.join()
        return ''.join(self.final_string)

