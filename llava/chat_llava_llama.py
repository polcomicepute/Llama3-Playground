from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler
import base64

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

# Replace 'file_path.png' with the actual path to your PNG file
file_path = '/home/jetson/cmap/athirdmapper/exp0610_ViT-B-16-SigLIP_3_copy/n_images/2.png'
data_uri = image_to_base64_data_uri(file_path)


# ../models/llava-mistral-gguf/mmproj-model-f16.gguf
# ../models/llava-llama/llava-llama-3-8b-v1_1-mmproj-f16.gguf
chat_handler = Llava16ChatHandler(clip_model_path='../models/llava-llama/llava-llama-3-8b-v1_1-mmproj-f16.gguf')

# '/home/jetson/llamaR/Llama3-Playground/models/llava-mistral-gguf/llava-v1.6-mistral-7b.Q4_K_M.gguf'
# ../models/llava-llama/llava-llama-3-8b-v1_1-int4.gguf'
llm = Llama(model_path='../models/llava-llama/llava-llama-3-8b-v1_1-int4.gguf',
            chat_handler=chat_handler,
            n_ctx = 4096,
            n_gpu_layers= 128)

few_shot_prompt = open('nlmap/obj_p.txt', 'r').read() 

user_input = input("원하는 Task를 입력하세요: ")
 
user_msg = f'''"""
<|start_header_id|>user<|end_header_id|>
You are the robot which is equipped with wheels and a manipulator arm, and the given image represents the environment you are observing.\n

you got the task `{user_input}`, identify the objects that could be involved.\n
Output should follow the format: `Objects: object1, object2, object3, object4` \n
    For example:\n
        Objects: table, napkin, sponge, towel\n
        Objects: fridge\n
        Objects: trash can for bottles\n
        Objects: human, candy, snickers, chips, apple, banana, orange\n

<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
"""'''

sys_msg='''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the robot which is equipped with wheels and a manipulator arm.\n

Here are some examples to guide you.\n 
''' 

response = llm.create_chat_completion(
    messages= [
        {"role" : "system", "content" : sys_msg + few_shot_prompt + "<|eot_id|>"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_msg},
                {"type": "image_url",  "image_url": {"url": data_uri }},
            ]   
        }
    ], 
)  
print(response["choices"][0]["message"]['content'])


'''

{'id': 'chatcmpl-8615da9b-f43d-4261-8e3c-38c2066aaf27', 'object': 'chat.completion', 'created': 1723044699, 'model': '/home/jetson/llamaR/Llama3-Playground/models/llava-mistral-gguf/llava-v1.6-mistral-7b.Q4_K_M.gguf', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '\n1. Based on the image, I am currently in a room with wooden cabinets and shelves. The room appears to be a storage or office area.\n2. From my current location, potential destinations for the robot could include:\na) A water bottle located near the top of one of the wooden shelves.\nb) A sink with a faucet, which is likely used for cleaning purposes.\nc) The refrigerator, which might contain additional bottles or containers of water.\n3. To execute the command \'Water, please\', the robot would need to:\na) Identify the location of the water source (either a bottle, sink, or refrigerator).\nb) Move towards that location using its wheels and manipulator arm.\nc) Once at the location, use its manipulator arm to either pick up a water bottle or open the refrigerator door to access stored water.\nd) Bring the water source back to the human who requested it.\n" '}, 'logprobs': None, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 3627, 'completion_tokens': 207, 'total_tokens': 3834}}

'''