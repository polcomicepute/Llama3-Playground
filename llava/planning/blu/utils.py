import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler
import torch
import gc 
llm = None
chat_handler = None

def initialize_llava():
    global llm, chat_handler
    chat_handler = Llava16ChatHandler(clip_model_path='../../../models/llava-mistral-gguf/mmproj-model-f16.gguf')
    llm = Llama(model_path='/home/jetson/llamaR/Llama3-Playground/models/llava-mistral-gguf/llava-v1.6-mistral-7b.Q4_K_M.gguf',
                chat_handler=chat_handler,
                n_ctx=5120,
                n_gpu_layers=128
                )

def initialize_llama():
    global llm
    llm = Llama(model_path='/home/jetson/llamaR/Llama3-Playground/models/llama_3_8b_instruct/ggml-model-Q4_K_M.gguf',
                chat_format="llama-3",
                n_ctx=2048,
                n_gpu_layers=128
                )


sys_msg='''
A chat between a curious human and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the human's questions.
You are an assistant who can describe images in great detail and the robot which is equipped with wheels and a manipulator arm.

Here are some examples to guide you.\n 
'''

sys_msg_llama='''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
A chat between a curious human and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the human's questions.
You are an assistant who can describe images in great detail and the robot which is equipped with wheels and a manipulator arm.

Here are some examples to guide you.\n 
'''
# Optionally, cleanup can be performed manually elsewhere in your code
def cleanup_llm():
    # GPU와 CPU 사이의 모든 작업을 동기화
    global llm, chat_handler
    print("state: ",llm is not None)
    if llm is not None:
        torch.cuda.synchronize()
        
        # 모델 및 핸들러 객체 삭제
        del llm
        del chat_handler 
        
        # 여러 번 가비지 컬렉션 실행
        for _ in range(3):
            gc.collect()
        
        # GPU 캐시 비우기
        torch.cuda.empty_cache()

        # 메모리 상태를 확인 (선택 사항)
        print(f'Memory allocated: {torch.cuda.memory_allocated()}')
        print(f'Memory reserved: {torch.cuda.memory_reserved()}') 
        llm = None
        chat_handler= None 

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

def open_file(file_path):
    with open(file_path, 'r') as file:
        few_shot_prompt = file.read()
        return few_shot_prompt
    

def chat_llava(user_msg, data_uri, few_shot_prompt=None,resonse_format=None):
    global llm
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


def chat_llama(user_msg, few_shot_prompt=None,resonse_format=None):
    global llm
    few_shot_prompt = few_shot_prompt if few_shot_prompt is not None else ""
    resonse_format = resonse_format if resonse_format is not None else ""
    response = llm.create_chat_completion(
    messages= [
        {"role" : "system", "content" : sys_msg_llama + few_shot_prompt + "<|eot_id|>"},
        {
            "role": "user",
            "content": user_msg
        }
    ], 
    temperature=0.0
    )  
    return response

