from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler 
from utils import * 
import re
from PIL import Image, ImageDraw


chat_handler = Llava16ChatHandler(clip_model_path='../../models/llava-mistral-gguf/mmproj-model-f16.gguf')
llm = Llama(model_path='/home/jetson/llamaR/Llama3-Playground/models/llava-mistral-gguf/llava-v1.6-mistral-7b.Q4_K_M.gguf',
            chat_handler=chat_handler,
            n_ctx=5120,
            n_gpu_layers=128
            )

with open('../nlmap/obj_p.txt', 'r') as file:
    plan_few_shot_prompt = file.read()

with open('../nlmap/plan_half.txt', 'r') as file:
    op_few_shot_prompt = file.read()


file_path = '/home/jetson/cmap/athirdmapper/exp0610_ViT-B-16-SigLIP_3_copy/n_images/2.png'
data_uri = image_to_base64_data_uri(file_path=file_path)

task = input("원하는 Task를 입력하세요: ")
 
op_user_msg = f'''"""
<image>\nUSER:\n
You are the robot which is equipped with wheels and a manipulator arm, and the given image represents the environment you are observing.\n

you got the task `{task}`, identify the objects that could be involved.\n
Output should follow the format: `Objects: object1, object2, object3, object4` \n
    For example:\n
        Objects: table, napkin, sponge, towel\n
        Objects: fridge\n
        Objects: trash can for bottles\n
        Objects: human, candy, snickers, chips, apple, banana, orange\n
.\nASSISTANT:\n
"""'''


op_response = llm.create_chat_completion(
    messages= [
        {"role" : "system", "content" : sys_msg + op_few_shot_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": op_user_msg},
                {"type": "image_url",  "image_url": {"url": data_uri }},
            ]   
        }
    ], 
)  
print(op_response["choices"][0]["message"]['content'])
avail_obj = op_response["choices"][0]["message"]['content']
avail_obj = avail_obj.replace("Objects: ", "")
 



od_user_msg = f"""'''
<image>\nUSER:\n
Detect {avail_obj} on the image. 


if (the object is detected): return Coordinates relative to image width and height. Output should follow the format: `[x_min, y_min, x_max, y_max]`.\n
else: Output should follow the format: `None`.\n\n
 
\nASSISTANT:\n
'''"""

od_response = llm.create_chat_completion(
    messages=[
        {"role": "user", 
         "content": [
             {"type": "text", "text": od_user_msg},
             {"type": "image_url", "image_url": {"url": data_uri}},
         ]
        }
    ]
)


print(od_response["choices"][0]["message"]["content"])
coords_string = od_response["choices"][0]["message"]["content"]
clean_string = re.sub(r'[^0-9.,]', '', coords_string)
coords_list = clean_string.split(",")


if 'None' in coords_list:
    od_res = 'None'
else:
    x_min, y_min, x_max, y_max = map(float, coords_list)
    print(x_min, y_min, x_max, y_max)

    image = Image.open(file_path)

    # Define the coordinates
    width, height = image.size
    coords = [x_min * width, y_min* height, x_max * width, y_max * height]

    # Draw the box
    draw = ImageDraw.Draw(image)
    draw.rectangle(coords, outline="red", width=3)

    # Save the modified image
    output_path = "./output/" + task +".png"
    image.save(output_path)






plan_user_msg = f"""'''
<image>\nUSER:\n
You are the robot which is equipped with wheels and a manipulator arm, and the given image represents the environment you are observing.\n

you got the task `{task}`, identify the objects that could be involved.\n
Available objects are `{avail_obj}`\n

Can you provide a concise, step-by-step plan for a robot to complete the following task using only the following actions: "find", "pick up", "go to", "put down", and "done"?

First, Please include an explanation before listing the steps, detailing what the robot should do overall with format 
Second, list only the essential steps using only the following actions: ["find", "pick up", "go to", "put down"] to complete the task, and make sure to end with "done" when the task is complete.
Output should follow the format:\n
    `Explanation:\n overall plan`
    `Robot: 
        1. task1
        2. task2
        3. task3
        \n
        `
        
\nASSISTANT:\n
'''"""
'''
And please provide a concise, step-by-step plan for a robot to complete the following task using only the following actions: "find", "pick up", "go to", "put down", and ending with "done" to signify the task is complete?


Can you provide step-by-step instructions for a robot to complete the following task?
Output should follow the format:
    Explanation: `description what to do\n
    Robot: `1. task1, 2. task2, 3. task3`
'''
plan_response = llm.create_chat_completion(
    messages= [
        {"role" : "system", "content" : sys_msg + plan_few_shot_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": plan_user_msg},
                {"type": "image_url",  "image_url": {"url": data_uri }},
            ]   
        }
    ], 
)  
print(plan_response["choices"][0]["message"]['content'])



'''

Task: Throw away a coffee cup
Available objects: coffee cup, trash can
Explanation: The user has requested me to throw away a coffee cup. Throwing away means putting something in the trash can. I will find a coffee cup, pick that up and then put it in the trash.
Robot:
1. find the coffee cup
2. pick up the coffee cup
3. go to the trash
4. put down the coffee cup
5. done

Task: Throw away the fruits
Available objects: apple, orange, banana, lime
Explanation: The user has requested me to throw away the fruits. Throwing away means putting something in the trash can. Banana is a type of fruit that’s available. I will find banana, pick that up and then put it in the trash.
Robot:
1. find the banana
2. pick up the banana
3. go to the trash
4. put down the banana
5. done
'''
