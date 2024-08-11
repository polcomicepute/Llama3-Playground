import base64

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
