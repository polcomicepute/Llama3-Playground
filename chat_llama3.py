from llama_cpp import Llama
llm = Llama(
      model_path="/home/jetson/llamaR/Llama3-Playground/models/llama_3_8b_instruct/ggml-model-Q4_K_M.gguf",
      chat_format="llama-3",
    n_ctx = 4096,
    n_gpu_layers= 128)
user_input = input("원하는 Task를 입력하세요: ")

few_shot_prompt = open('llava/nlmap/obj_p_half.txt', 'r').read()






'''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''






user_msg = f'''"""
<|start_header_id|>user<|end_header_id|>
You are the robot which is equipped with wheels and a manipulator arm.\n



you got the task `{user_input}`, identify the objects that may be involved.\n
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
      messages = [
          {"role" : "system", "content" : sys_msg + few_shot_prompt + "<|eot_id|>"},
          {
              "role": "user",
              "content": user_msg,
          }
      ]
)

print(response["choices"][0]["message"]['content'])