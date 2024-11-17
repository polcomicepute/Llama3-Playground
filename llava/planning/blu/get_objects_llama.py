from utils import * 
import re
from PIL import Image, ImageDraw 
# from robot_function import * 
import gc 


obj_proposal_few_shot = open_file('../../nlmap/obj_p.txt')

# if use llama3-8b-instruct (no image)
initialize_llama()

# if use llava-next (image O)
# initialize_llava()

task = input("원하는 Task를 입력하세요: ")

#! object_proposal 
op_user_msg = f'''"""
<|start_header_id|>user<|end_header_id|>
You are the robot which is equipped with wheels and a manipulator arm, and the given image represents the environment you are observing.\n

you got the task `{task}`, identify the objects that could be involved.\n
Output should follow the format: `Objects: object1, object2, object3, object4` \n
    For example:\n
        Objects: table, napkin, sponge, towel\n
        Objects: fridge\n
        Objects: trash can for bottles\n
        Objects: human, candy, snickers, chips, apple, banana, orange\n
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
"""'''

op_response = chat_llama(user_msg=op_user_msg, few_shot_prompt=obj_proposal_few_shot)
print(op_response["choices"][0]["message"]['content'])
avail_obj = op_response["choices"][0]["message"]['content']
avail_obj = avail_obj.replace("Objects: ", "") 



torch.cuda.empty_cache()

'''
원하는 Task를 입력하세요: give me a coke

llama_print_timings:        load time =    4171.24 ms
llama_print_timings:      sample time =      10.84 ms /     6 runs   (    1.81 ms per token,   553.71 tokens per second)
llama_print_timings: prompt eval time =    5780.83 ms /   693 tokens (    8.34 ms per token,   119.88 tokens per second)
llama_print_timings:        eval time =    1169.46 ms /     5 runs   (  233.89 ms per token,     4.28 tokens per second)
llama_print_timings:       total time =    7061.90 ms /   698 tokens
Objects: coke, fridge



'''