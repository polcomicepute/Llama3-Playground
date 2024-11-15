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
file_path = './images.png' # dog_bike_car
data_uri = image_to_base64_data_uri(file_path=file_path)


chat_handler = Llava16ChatHandler(clip_model_path='../../models/llava-mistral-gguf/mmproj-model-f16.gguf')
llm = Llama(model_path='/home/jetson/llamaR/Llama3-Playground/models/llava-mistral-gguf/llava-v1.6-mistral-7b.Q4_K_M.gguf',
            chat_handler=chat_handler,
            n_ctx=4096,
            n_gpu_layers=128
            )

# with open('nlmap/plan.txt', 'r') as file:
#     few_shot_prompt = file.read()

user_input = input("원하는 odj를 입력하세요: ")


user_msg = f"""
<image>\nUSER:\n
First, please provide chain-of-thought(cot) that includes the following steps: 
1. Observation: Analyze all the objects present in the image.
2. Thought: Determine whether the objects from the user's list [{user_input}] are present or not in the image. Think based on the observation(at point 1). If the object was not said at the observation, then it is not present in the image.


Second, based on the cot, please detect 'only' [{user_input}] in the image.
    For each object in the list [{user_input}],
    for obj in [{user_input}]:
	    if (obj is present), return the coordinates in the following format: 
		    {{
		        "detected" : true
		        "object_name" : obj,
		        "x_min": <float>,
		        "y_min": <float>,
		        "x_max": <float>,
		        "y_max": <float>
		    }}\n
	    else(obj is not present), return the coordinates in the following format: 
		    {{
		        "detected" : false
		        "object_name" :obj,
		        "x_min": -1000,
		        "y_min": -1000,
		        "x_max": -1000,
		        "y_max": -1000
		    }} 
As a result, return the results in the following valid JSON format:
    {{
        "cot": {{
            "observation": "<Analysis of all the objects present in the image>",
            "thought": "<Determine of whether the objects from the user's list are present in the image based on observation>",
        }},
        "detected_objects": [
            {{
                "detected": <boolean>,
                "object_name": "<object_name>",
                "x_min": <number>,
                "y_min": <number>,
                "x_max": <number>,
                "y_max": <number>
            }},
            ...
        ]
    }}
\nASSISTANT:\n
"""
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
    response_format = {
        "type": "object",
        "properties": {
            "cot": {
                "type": "array",
                "description":  "whether specific objects are present in the image or not",
                "items": {
                    "type": "object", 
                    "properties": {
                        "observation": {"type": "string"},
                        "thought": {"type": "string"},
                    },
                    "required": ["observation", "thought"]
                }
            },
            "detected_objects": {
                "type": "array",
                "description": "A dictionary of detected objects based on explanation and their coordinates.",
                "items": {
                    "type": "object",
                    "properties": {
                        "detected": {"type": "boolean"},
                        "object_name": {"type": "string"},
                        "x_min": {"type": "number"},
                        "y_min": {"type": "number"},
                        "x_max": {"type": "number"},
                        "y_max": {"type": "number"}
                    },
                    "required": ["detected", "object_name", "x_min", "y_min", "x_max", "y_max"]
                }
            }
        },
        "required": ["cot", "detected_objects"]
    },
    temperature = 0.0
)
'''
3. Reasoning: For each object from the user's list [{user_input}], clearly state whether it is present or absent based on the thought(at point 2). If present, provide a brief description of its location and appearance. If absent, clearly state that it is not present.
Ensure that the cot covers both points 1,2 and 3.


response_format={
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "country": {"type": "string"},
            "capital": {"type": "string"},
        },
        "required": ["country", "capital"],
    },
}

                {
                    "type": "object",
                    "properties": {
                        "None": {"type": "boolean"}
                    },
                    "required": ["None"]
                }
'''
print(response["choices"][0]["message"])
print(response["choices"][0]["message"]["content"])
coords_string = response["choices"][0]["message"]["content"]
# 문자열에서 대괄호와 공백을 제거하고, 쉼표로 구분하여 리스트로 변환
# coords_list = coords_string.strip("\n").strip(" ").strip("[]").split(",")
 
 
coords_string = re.sub('```', '', coords_string)
print(coords_string)
coords_string = re.sub('json', '', coords_string)

print(coords_string)
data = json.loads(coords_string)
# # 쉼표로 구분하여 리스트로 변환
# coords_list = clean_string.split(",")


# data = json.loads(coords_string)

image = Image.open(file_path)
width, height = image.size

od_res = []

for obj in data["detected_objects"]:
    print(obj)
    if obj["detected"] == False:
        continue
    else:
        # 모든 객체에 대해 바운딩 박스를 그리기 
        x_min, y_min, x_max, y_max = map(float, [obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']])
        
        # 좌표 계산
        box_coords = [x_min * width, y_min * height, x_max * width, y_max * height]
        
        # 바운딩 박스 그리기
        draw = ImageDraw.Draw(image)
        draw.rectangle(box_coords, outline="red", width=3)
        od_res.append(obj["object_name"])

# Save the modified image
output_path = "./output_2/" + user_input +".png"
image.save(output_path)



print("최종 탐지 결과:", od_res)











'''
"schema": {
    "type": "object",
    "properties": {"team_name": {"type": "string"}},  # team_name 속성이 문자열 타입이어야 함
    "required": ["team_name"],  # team_name 속성이 필수임
},
response_format={
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "country": {"type": "string"},
            "capital": {"type": "string"},
        },
        "required": ["country", "capital"],
    },
}




response_format={
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "published_date": {"type": "string"},
            "isbn": {"type": "string"}
        },
        "required": ["title", "author"]
    }
}
{
  "title": "The Great Gatsby",
  "author": "F. Scott Fitzgerald",
  "published_date": "1925",
  "isbn": "9780743273565"
}


response_format={
    "type": "json_array",
    "items": {
        "type": "string"
    }
} ["apple", "banana", "orange"]


'''






    # x_min, y_min, x_max, y_max = map(float, coords_list)
    # print(x_min, y_min, x_max, y_max)

    # image = Image.open(file_path)

    # # Define the coordinates
    # width, height = image.size
    # coords = [x_min * width, y_min* height, x_max * width, y_max * height]

    # # Draw the box
    # draw = ImageDraw.Draw(image)
    # draw.rectangle(coords, outline="red", width=3)

