from llama_cpp import Llama

class llmResponse():
    MODEL_PATH = "./models/Llama-3-Instruct-8B-SPPO-Iter3-IQ4_XS.gguf"
    SYS_PROMPT = "You are a helpful, respectful, and honest AI Assistant that can speak out your responses. Please always respond as though you are chatting with the user."

    def __init__(self, path = MODEL_PATH) -> None:
        self.model = Llama(model_path=path,chat_format="llama-3")
        self.prompt = ""
        self.messages = []
    
    def return_message_temp (self, prompt:str, sysPrompt = SYS_PROMPT) :
        self.prompt = prompt
        msg_temp = [{"role": "system", "content": sysPrompt },
                    {"role": "user", "content": self.prompt}]
        return msg_temp
    
    def create_response(self, msg:str):
        response = self.model.create_chat_completion(messages = msg, max_tokens=512, temperature=0.5, top_p=0.95,
                repeat_penalty=1.2, top_k=150)

        clean_response = response["choices"][0]['message']['content']
        self.messages.append(clean_response)
        return clean_response


