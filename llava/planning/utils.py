import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler

chat_handler = Llava16ChatHandler(clip_model_path='../../models/llava-mistral-gguf/mmproj-model-f16.gguf')
llm = Llama(model_path='/home/jetson/llamaR/Llama3-Playground/models/llava-mistral-gguf/llava-v1.6-mistral-7b.Q4_K_M.gguf',
            chat_handler=chat_handler,
            n_ctx=5120,
            n_gpu_layers=128
            )

sys_msg='''
A chat between a curious human and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the human's questions.
You are an assistant who can describe images in great detail and the robot which is equipped with wheels and a manipulator arm.

Here are some examples to guide you.\n 
'''


def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

def open_file(file_path):
    with open(file_path, 'r') as file:
        few_shot_prompt = file.read()
        return few_shot_prompt
    

def chat(user_msg, data_uri, few_shot_prompt=None,resonse_format=None):
    few_shot_prompt = few_shot_prompt if few_shot_prompt is not None else ""
    resonse_format = resonse_format if resonse_format is not None else ""
    response = llm.create_chat_completion(
    messages= [
        {"role" : "system", "content" : sys_msg + few_shot_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_msg},
                {"type": "image_url",  "image_url": {"url": data_uri }},
            ]   
        }
    ], 
    temperature=0.0
    )  
    return response
