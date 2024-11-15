from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler
import base64
from PIL import Image, ImageDraw
import re
import json

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

# Replace 'file_path.png' with the actual path to your PNG file
# img = input("원하는 사진 이름 입력: ")
# file_path = './' + img #'./404.png'  
file_path = './test.png' 
data_uri = image_to_base64_data_uri(file_path=file_path)


chat_handler = Llava16ChatHandler(clip_model_path='../../models/llava-mistral-gguf/mmproj-model-f16.gguf')
llm = Llama(model_path='/home/jetson/llamaR/Llama3-Playground/models/llava-mistral-gguf/llava-v1.6-mistral-7b.Q4_K_M.gguf',
            chat_handler=chat_handler,
            n_ctx=4096,
            n_gpu_layers=128
            )


user_msg = f"""
<image>\nUSER:\n

You are a traffic light detector.  
You are currently in a road driving scenario. 
Traffic lights are usually located above the lanes.


Identify the traffic light state corresponding to your lane.
Note: Available traffic light states are (red, green, yellow, none), If unsure, say "none". 
Note: Ignore the lights on the cars.  



\nASSISTANT:\n
"""
#You are a lane localizer.  
#You are currently in a road driving scenario, where the black area represents the ground, and the white lines represent the lanes.  
#Answer which lane you are currently in, counting from the far left. 

# corresponding to your lane.

print(user_msg)

  

response = llm.create_chat_completion(
    messages=[
        {"role": "user", 
         "content": [
             {"type": "text", "text": user_msg},
             {"type": "image_url", "image_url": {"url": data_uri}},
         ]
        }
    ],
    temperature = 0.0
)
 
print(response["choices"][0]["message"])
print(response["choices"][0]["message"]["content"])
 



# print("최종 탐지 결과:", od_res)





 
