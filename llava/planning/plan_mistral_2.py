
from utils import *
import re
from PIL import Image, ImageDraw


plan_few_shot_prompt = open_file('../nlmap/obj_p.txt') 
op_few_shot_prompt = open_file('../nlmap/plan_half.txt')

data_uri = image_to_base64_data_uri(file_path='/home/jetson/cmap/athirdmapper/exp0610_ViT-B-16-SigLIP_3_copy/n_images/2.png')


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

op_response = chat(user_msg=op_user_msg, data_uri=data_uri, few_shot_prompt=op_few_shot_prompt)

print(op_response["choices"][0]["message"]['content'])
avail_obj = op_response["choices"][0]["message"]['content']
avail_obj = avail_obj.replace("Objects: ", "")

 

#! planning
plan_user_msg = f"""
<image>\nUSER:\n
You are the robot which is equipped with wheels and a manipulator arm, and the given image represents the environment you are observing.\n

you got the task `{task}`, identify the objects that could be involved.\n
Available objects are `{avail_obj}`\n

Can you provide a concise, step-by-step plan for a robot to complete the following task using only the following action functions: "find", "pick_up", "go_to", "put_down", and "done"?
The explanation of the action functions are as follow:
	def go_to(coordinate: string):
		# navigate robot base to location coordinates
		# location can be object, location's, or description's coordinate

	def find(object: string):
		# find object location coordinate using camera equipped in robot arm
		return ([x, y, z] float coordinate of object)
		
	def pick_up(what_to: string):
		# move robot arm to 'what_to' coordinate and grasp
		
	def put_down(where_to: string):
		# move robot arm to 'where_to' coordinate and release 

	def done : terminate when the plans are done or plan is nothing


First, Please include an explanation before listing the steps, detailing what the robot should do overall with format 
Second, list only the essential steps using only the following action functions: ["find", "pick_up", "go_to", "put_down"] to complete the task, and make sure to end with "done" when the task is complete.
Output should follow the format:\n
Explanation:\n overall plan
Robot: 
1. task1on, it should navigate to that location using the "go_to" action function
2. task2
3. task3

for example, 
given available objects: coffee cup, trash can 
Explaination:
The robot should first "find" the coffee cup's location, which is likely to be located near a table or counter where people usually drink coffee. Once the robot has found the coffee cup, it should "go_to" the coffee cup's location. Pick up the cup using its manipulator arm from the table or counter. Next, "find" the trash can's location and "go_to" the trash can. After reaching the trash can, the robot should "put_down" the coffee cup inside the trash can. Finally, the robot should confirm that the task is complete by saying "done".
Robot:  
2. go_to(find(coffee cup))
2. pick_up(coffee cup)
3. go_to(find(trash can))
4. put_down(coffee cup, the trash can)
5. done 
\n

        
\nASSISTANT:\n""" 

plan_response = chat(user_msg=plan_user_msg, data_uri=data_uri, few_shot_prompt=None)
print(plan_response["choices"][0]["message"]['content'])
