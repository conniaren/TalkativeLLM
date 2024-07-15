from llama_cpp import Llama

MODEL_PATH = "./models/Llama-3-Instruct-8B-SPPO-Iter3-IQ4_XS.gguf"
PROMPT = "Hello, How are you ? My name is Connie Ren."
MESSAGES = []

def initialize_model():
    llm = Llama(
        model_path=MODEL_PATH,
        chat_format="llama-3"
    )
    return llm

def get_llama_response(model, prompt=PROMPT):

    system_prompt = "You are a helpful, respectful, and honest AI Assistant that can speak out your responses. Please always respond as though you are chatting with the user."

   # prompt_template= f"<|start_header_id|>system<|end_header_id|>
   # { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>
   # { prompt }<|eot_id|><|start_header_id|>assistant<|end_header_id|> "

    msg=[{"role": "system", "content": system_prompt},
         {"role": "user", "content": prompt}]
    

    MESSAGES.append(msg)
    response = model.create_chat_completion(messages = msg, max_tokens=256, temperature=0.5, top_p=0.95,
                repeat_penalty=1.2, top_k=150)

    clean_response = response["choices"][0]['message']['content']
    print(clean_response)
    return clean_response

if __name__ == "__main__":
    model = initialize_model()
    get_llama_response(model)
    get_llama_response(model, prompt="I just wanted to ask for help, what do I do during hard times? No job, no friends etc.")