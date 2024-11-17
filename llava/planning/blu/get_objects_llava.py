from utils import * 
import re
from PIL import Image, ImageDraw 
# from robot_function import * 
import gc 
 
obj_proposal_few_shot = open_file('../../nlmap/obj_p.txt')

        
data_uri = image_to_base64_data_uri(file_path='/home/jetson/cmap/athirdmapper/exp0610_ViT-B-16-SigLIP_3_copy/n_images/2.png')


# if use llava-next (image O)
initialize_llava() 

task = input("원하는 Task를 입력하세요: ")

#! object_proposal
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

op_response = chat_llava(user_msg=op_user_msg, data_uri=data_uri, few_shot_prompt=obj_proposal_few_shot)

print(op_response["choices"][0]["message"]['content'])
avail_obj = op_response["choices"][0]["message"]['content']
avail_obj = avail_obj.replace("Objects: ", "") 



torch.cuda.empty_cache()


'''
원하는 Task를 입력하세요: give me a coke
encode_image_with_clip: 5 segments encoded in  2660.82 ms
encode_image_with_clip: image embedding created: 2880 tokens

encode_image_with_clip: image encoded in  3108.49 ms by CLIP (    1.08 ms per image patch)
Llama.generate: prefix-match hit

llama_print_timings:        load time =   98245.23 ms
llama_print_timings:      sample time =       5.46 ms /    11 runs   (    0.50 ms per token,  2012.81 tokens per second)
llama_print_timings: prompt eval time =       0.00 ms /     0 tokens (     nan ms per token,      nan tokens per second)
llama_print_timings:        eval time =    2198.40 ms /    11 runs   (  199.85 ms per token,     5.00 tokens per second)
llama_print_timings:       total time =    2258.97 ms /    11 tokens


Objects: table, coke bottle 
'''
