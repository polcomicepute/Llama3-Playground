import re
import dmap
 
# Pick up the pen from the desk and place it in the drawer 
# Move the water bottle from the desk to the fridge
# Grab the coffee mug and place it in the sink
# Move the chair closer to the conference table
# Grab the hand sanitizer and place it near the door

def execute_command(commands):
    # 맵핑 객체 초기화 (이미 존재한다고 가정)
    map = dmap.DMAPNode(predefined=False, debug=True)  
    
    for command in commands:
        command = command.strip()

        if command.startswith("go_to"):
            # 위치와 대체 위치 후보를 추출
            match = re.search(r"go_to\(([^,]+)(?:,\s*\[(.*)\])?\)", command)
            if match:
                location = match.group(1).strip()
                location_candidates = match.group(2)
                if location_candidates:
                    location_candidates = [loc.strip() for loc in location_candidates.split(',')]
                else:
                    location_candidates = None
                
                # go_to 함수 호출
                go_to(location=location, location_candidate=location_candidates, map=map)
        
        elif command.startswith("pick_up"):
            # pick_up 명령어에서 객체 추출
            object_name = re.search(r"pick_up\((.+)\)", command).group(1).strip()
            pick_up(object_name)
        
        elif command.startswith("put_down"):
            # put_down 명령어에서 객체와 위치 추출
            match = re.search(r"put_down\((.+),\s*(.+)\)", command)
            if match:
                object_name = match.group(1).strip()
                location = match.group(2).strip()
                put_down(object_name, location)
        
        elif command == "done":
            # 모든 작업 완료를 확인
            print("All tasks completed!")
        
        else:
            # 알 수 없는 명령어 처리
            print(f"Unknown command: {command}")

   

def parse_cmd(command: str):
    # "Robot:" 이후의 명령어 부분을 추출
    command_list = command.split("Robot:")[-1]
    
    # 개별 명령어를 추출 (예: "1. go_to(...)", "2. pick_up(...)" 등)
    commands = re.findall(r'\d+\.\s*(.+)', command_list.strip())
    
    # 디버깅용 출력문
    print(f"Extracted commands: {commands}")

    return commands

  
def go_to(location, location_candidate, map):
    print(f"Navigating to {location}, candidates: {location_candidate}") 
    map.get_goal(location)
    # 여기에 위치 이동 로직 구현
    pass

def pick_up(object):
    print(f"Picking up {object}")
    # 여기에 객체 집기 로직 구현
    pass

def put_down(object, location):
    print(f"Putting down {object} in {location}")
    # 여기에 객체 내려놓기 로직 구현
    pass

plan_response = '''
Explanation:
The robot should first find the notepad's location and go to the notepad using "go_to" function. Notepad's location is likely to be located near a table or counter where people usually write notes. 
Once the robot has found the notepad, it should pick up the notepad using its manipulator arm from the table or counter. 
Next, find the robot's base location and go to the robot's base using "go_to" function. After reaching the robot's base, the robot should "put_down" the notepad inside the robot's base. Finally, the robot should confirm that the task is complete by saying "done".
Robot:  
1. go_to(notepad)
2. pick_up(notepad)
3. go_to(robot's base)
4. put_down(notepad, robot's base)
5. done 
'''
'''
Explanation:
The robot should first find the location of the fruits using "go_to" function. Once it has found the fruits, it should pick up each fruit one by one using its manipulator arm and put them down in a trash can for bottles. Finally, the robot should confirm that the task is complete by saying "done".
Robot:  
1. go_to(fruits)
2. pick_up(apple)
3. go_to(trash can for bottles)
4. put_down(apple, trash can for bottles)
5. pick_up(banana)
6. go_to(trash can for bottles)
7. put_down(banana, trash can for bottles)
8. done
'''
# commands = parse_cmd(plan_response)
# for command in commands:
#     execute_command(command.strip())
'''

    # query location
    # map.get_goal(location)
    # if evaluate find location candidiates 

    # navigate


def query_llm(query):
    result = send_query_to_llm(query)  # LLM에게 쿼리 보내기
    if evaluate_result(result):  # 결과 평가
        return result
    else:
        return fallback_action()  # 다른 행동 실행

def send_query_to_llm(query):
    # LLM에 쿼리 보내는 함수
    pass

def evaluate_result(result):
    # 결과를 평가하는 함수, 예를 들어 결과가 빈 경우 False 반환
    pass

def fallback_action():
    # 대체 행동, 예를 들어 다른 쿼리나 다른 API 호출
    pass



# 기존 함수 정의
def go_to(location):
    print(f"Navigating to {location}")
    # 여기에 위치 이동 로직 구현
    pass

def pick_up(object):
    print(f"Picking up {object}")
    # 여기에 객체 집기 로직 구현
    pass

def put_down(object, location):
    print(f"Putting down {object} in {location}")
    # 여기에 객체 내려놓기 로직 구현
    pass

# 전체 명령어 문자열
command_string = """
Explanation:
The robot should first find the location of the fruits using "go_to" function. Once it has found the fruits, it should pick up each fruit one by one using its manipulator arm and put them down in a trash can for bottles. Finally, the robot should confirm that the task is complete by saying "done".
Robot: 
1. go_to(fruits)
2. pick_up(apple)
3. go_to(trash can for bottles)
4. put_down(apple, trash can for bottles)
5. pick_up(banana)
6. go_to(trash can for bottles)
7. put_down(banana, trash can for bottles)
8. done 
"""

# 명령어 부분만 추출 (Robot: 이후의 부분만)
command_list = command_string.split("Robot:")[-1].strip().split("\n")

# 명령어 매핑을 위한 함수
def execute_command(command):
    parts = command.split()
    
    if parts[0] == "go_to":
        location = command[7:-1]  # go_to( 이후와 ) 이전의 내용 추출
        go_to(location)
    elif parts[0] == "pick_up":
        object = command[9:-1]  # pick_up( 이후와 ) 이전의 내용 추출
        pick_up(object)
    elif parts[0] == "put_down":
        object, location = command[9:-1].split(",")  # put_down( 이후 내용 추출
        object = object.strip()  # 객체 이름 정리
        location = location.strip()  # 위치 이름 정리
        put_down(object, location)
    elif parts[0] == "done":
        print("All tasks completed!")
    else:
        print(f"Unknown command: {command}")

# 명령어를 순차적으로 실행
for command in command_list:
    cleaned_command = command.strip()
    execute_command(cleaned_command)

    
    -----



import re

# 기존 함수 정의
def go_to(location):
    print(f"Navigating to {location}")
    # 로봇이 해당 위치로 이동하는 로직 구현
    pass

def pick_up(object):
    print(f"Picking up {object}")
    # 로봇이 객체를 집는 로직 구현
    pass

def put_down(object, location):
    print(f"Putting down {object} in {location}")
    # 로봇이 객체를 특정 위치에 내려놓는 로직 구현
    pass

# 명령어 매핑을 위한 파싱 및 실행 함수
def execute_command(command):
    if command.startswith("go_to"):
        location = re.search(r"go_to\((.+)\)", command).group(1)
        go_to(location.strip())
    elif command.startswith("pick_up"):
        object = re.search(r"pick_up\((.+)\)", command).group(1)
        pick_up(object.strip())
    elif command.startswith("put_down"):
        match = re.search(r"put_down\((.+),\s*(.+)\)", command)
        object = match.group(1).strip()
        location = match.group(2).strip()
        put_down(object, location)
    elif command == "done":
        print("All tasks completed!")
    else:
        print(f"Unknown command: {command}")

# 전체 명령어 문자열
commands_str = """
1. go_to(fruits)
2. pick_up(apple)
3. go_to(trash can for bottles)
4. put_down(apple, trash can for bottles)
5. pick_up(banana)
6. go_to(trash can for bottles)
7. put_down(banana, trash can for bottles)
8. done
"""

# 명령어 추출
def parse_commands(commands_str):
    # 정규 표현식을 사용하여 명령어 리스트 추출
    commands = re.findall(r'\d+\.\s*(.+)', commands_str)
    return commands

# 명령어 파싱 및 실행
commands = parse_commands(commands_str)
for command in commands:
    execute_command(command.strip())

'''
