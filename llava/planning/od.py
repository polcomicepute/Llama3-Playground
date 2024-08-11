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
file_path = './2005.png'
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
Please detect only the following objects in the image: [{user_input}].

For each object in the list [{user_input}],
    for i, obj in enumerate([{user_input}]):
        if (obj is detected), return the coordinates in the following format:
            {{
                i: {{
                    "detected" : True
                    "object_name" : obj,
                    "x_min": <float>,
                    "y_min": <float>,
                    "x_max": <float>,
                    "y_max": <float>
                }}
            }}\n
        else, return the coordinates in the following format:
            {{
                i: {{
                    "detected" : False
                    "object_name" :obj,
                    "x_min": -1000,
                    "y_min": -1000,
                    "x_max": -1000,
                    "y_max": -1000
                }}
            }}\n

Ensure that you only return results for the objects listed in [{user_input}], and no others. The output must be in valid JSON format.
\nASSISTANT:\n
"""




# return (Just Coordinates defined in [x_min, y_min, x_max, y_max] relative to image width and height)\n\n
#     Output should follow the format: `[x_min, y_min, x_max, y_max]`\n 
# If (the object is NOT detected): 

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
        "type": "json_object",
        "patternProperties": {
                "^[0-9]+$": {
                    "type": "object",
                    "properties": { 
                            "detected" : {"type": "boolean"},
                            "object_name" : {"type": "string"},
                            "x_min": {"type": "number"},
                            "y_min": {"type": "number"},
                            "x_max": {"type": "number"},
                            "y_max": {"type": "number"}
                            },
                    "required": ["detected", "object_name", "x_min", "y_min", "x_max", "y_max"],
        }
    }}
)
'''
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

print(response["choices"][0]["message"]["content"])
coords_string = response["choices"][0]["message"]["content"]
# 문자열에서 대괄호와 공백을 제거하고, 쉼표로 구분하여 리스트로 변환
# coords_list = coords_string.strip("\n").strip(" ").strip("[]").split(",")
data = json.loads(coords_string)
 
# clean_string = re.sub(r'[^0-9.,]', '', coords_string)
# # 쉼표로 구분하여 리스트로 변환
# coords_list = clean_string.split(",")


data = json.loads(coords_string)

image = Image.open(file_path)
width, height = image.size

od_res = []

for i, obj in data.items():
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
output_path = "./output/" + user_input +".png"
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

