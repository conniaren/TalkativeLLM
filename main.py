import src.RecordSTT as RecordSTT
import src.LLMResponse as LLMResponse
import src.OutputTTS as OutputTTS
import keyboard
import time 
from threading import Thread
import os
import sys


def main():
    stt = RecordSTT.recordSTT()
    llm = LLMResponse.llmResponse()
    tts = OutputTTS.outputTTS()
    os.system('clear')

    print("This is your generative AI assistant. Press enter to start speaking to it: \n")

    global is_running
    is_running = True

    def check_escape():
        global is_running
        while is_running:
            if keyboard.is_pressed('esc'):
                is_running = False
                print("Escape key pressed. Exiting the program.")
                time.sleep(0.1)
                break         
        sys.exit()

    while True:
        if keyboard.is_pressed('enter'):
            time.sleep(0.2)
            try:
                user_prompt = stt.start_recording()
            except Exception as e:
                print(f"Error recording audio content: {e}")
                break

            try:
                template = llm.return_message_temp(user_prompt)
                model_response = llm.create_response(template)
            except Exception as e:
                print(f"Error generating model response: {e}")
                break
            
            try:
                audio_waves = tts.run_inference(model_response)
                tts.play_audio(audio=audio_waves)
            except Exception as e:
                print(f"Error playing model audio: {e}")
                break


            print("Press enter to speak again. Or press ESC to exit the program. \n")
            escape_thread = Thread(target=check_escape)
            escape_thread.start()
            while keyboard.is_pressed('enter'):
                pass

if __name__ == '__main__':
    main()