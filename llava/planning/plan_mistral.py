from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler
import base64
from PIL import Image, ImageDraw
import re
def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

# Replace 'file_path.png' with the actual path to your PNG file
img = input("원하는 사진 이름 입력: ")
file_path = './' + img #'./404.png'
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

user_msg = f"""'''
<image>\nUSER:\n
Detect {user_input} on the image. 


if (the object is detected): return Coordinates relative to image width and height. Output should follow the format: `[x_min, y_min, x_max, y_max]`.\n
else: Output should follow the format: `None`.\n\n

    
     

\nASSISTANT:\n
'''"""
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
    ]
)

print(response["choices"][0]["message"]["content"])
coords_string = response["choices"][0]["message"]["content"]
# 문자열에서 대괄호와 공백을 제거하고, 쉼표로 구분하여 리스트로 변환
# coords_list = coords_string.strip("\n").strip(" ").strip("[]").split(",")
clean_string = re.sub(r'[^0-9.,]', '', coords_string)
# 쉼표로 구분하여 리스트로 변환
coords_list = clean_string.split(",")

# 각 리스트 요소를 float 타입으로 변환
x_min, y_min, x_max, y_max = map(float, coords_list)

# 변환된 값 출력
print(x_min, y_min, x_max, y_max)

image = Image.open(file_path)

# Define the coordinates
width, height = image.size
coords = [x_min * width, y_min* height, x_max * width, y_max * height]

# Draw the box
draw = ImageDraw.Draw(image)
draw.rectangle(coords, outline="red", width=3)

# Save the modified image
output_path = "./output/" + user_input +".png"
image.save(output_path)

output_path