import json
import os
import numpy as np
import random

HOW_MANY_SKILLS = 20
assert HOW_MANY_SKILLS % 5 == 0
skills_per_length = HOW_MANY_SKILLS // 5

def process_skill_strings(strings):
    processed_strings = []
    for string in strings:
        string = string.strip().lower()
        if string[-1] not in ["?", ".", "!"]:
            string = string + "."
        processed_strings.append(string.capitalize())
    return processed_strings




def generate_skill(annotations, sample_ids, traj_data, repeat_id, scene_data, name):
    generated_skill = dict()
    traj_skill = traj_data["plan"]["high_pddl"]
    file_name = traj_data["task"]
    for i, skill in enumerate(traj_skill):
        if "high_idx" not in skill:
            skill["high_idx"] = i
    matching_skills = [skill for skill in traj_skill if skill["high_idx"] in sample_ids]
    for scene_index in range(len(scene_data)):
        scene = scene_data[scene_index]
        if scene["random_seed"] == traj_data["scene"]["random_seed"]:
            use_scene_index = scene_index
            break

    for skill_index in range(len(matching_skills)):
        skill = matching_skills[skill_index]
        high_idx = skill["high_idx"]
        api_action = [
            action["discrete_action"]["action"]
            for action in traj_data["plan"]["low_actions"]
            if action["high_idx"] == high_idx
        ]
        skill.pop("high_idx")
        skill["api_action"] = api_action
        skill["scene_index"] = use_scene_index

        skill["annotations"] = traj_data["turk_annotations"]["anns"][repeat_id]["high_descs"][skill_index]

    
    generated_skill["task"] = name
    generated_skill["repeat_id"] = repeat_id
    generated_skill["subgoal_ids"] = list(sample_ids)
    generated_skill["primitive_skills"] = matching_skills
    generated_skill["annotation"] = annotations
    return generated_skill




scene_type = "valid_unseen"
scene_path = f"../scene_sampling/{scene_type}_scene.json"
with open(scene_path, "r") as f:
    scene_data = json.load(f)

sample_tasks_path = f"../data/splits/oct21.json"
with open(sample_tasks_path, "r") as f:
    sample_tasks = json.load(f)[scene_type]


dataset_path = "../data/json_2.1.0_merge_goto/" + scene_type + "/"

dict_num = dict()
dict_name = dict()

for i in (sample_tasks):
    name = i["task"]
    path = os.path.join(dataset_path, name, "augmented_traj_data_new.json")
    with open(path, "r") as f:
        traj_data = json.load(f)

    skills = traj_data["plan"]["high_pddl"]
    if skills[-1]["discrete_action"]["action"] == "NoOp":
        skills = skills[:-1]

    num_skill = len(skills)
    if num_skill not in dict_num:
        dict_num[num_skill] = 0
        dict_name[num_skill] = []
    dict_num[num_skill] += 1
    dict_name[num_skill].append(name)

new_dict = dict()
for i, j in dict_name.items():
    print("before",i, len(j))
    j = list(set(j))
    new_dict[i] = j

for i, j in new_dict.items():
    print("after",i, len(j))
dump_list = []

primitive_sample_num = 0

longer_traj_list = []

# sample skills from each length
for skill_nums, run_names in new_dict.items():
    
    if skill_nums in [2,3,4,6]: # 6 because no length 5 skills
        num = len(run_names)
        indices = random.sample(range(num), skills_per_length)

        for index in indices:
            use_name = run_names[index]
            path = os.path.join(dataset_path, use_name, "augmented_traj_data_new.json")
            with open(path, "r") as f:
                traj_data = json.load(f)
            
            num_annotations = len(traj_data["turk_annotations"]["anns"])
            repeat_id = random.sample(range(num_annotations), 1)[0]

            annotation = traj_data["turk_annotations"]["anns"][repeat_id]["task_desc"]
            sample_ids = list(range(skill_nums))
            print(skill_nums, sample_ids)
            output_skill = generate_skill(annotation, sample_ids, traj_data, repeat_id, scene_data, use_name)
            dump_list.append(output_skill)

    else:
        longer_traj_list.extend(run_names)

# sample primitive skills
num = len(longer_traj_list)
indices = random.sample(range(num), skills_per_length)
for index in indices:
    use_name = longer_traj_list[index]
    path = os.path.join(dataset_path, use_name, "augmented_traj_data_new.json")
    with open(path, "r") as f:
        traj_data = json.load(f)
    
    num_annotations = len(traj_data["turk_annotations"]["anns"])
    repeat_id = random.sample(range(num_annotations), 1)[0]

    ramdon_skill_num = random.sample(range(num_annotations),1)[0]

    annotation = traj_data["turk_annotations"]["anns"][repeat_id]["high_descs"][ramdon_skill_num]
    sample_ids = [ramdon_skill_num]
#     sample_ids = list(range(skill_nums - 1))
    output_skill = generate_skill(annotation, sample_ids, traj_data, repeat_id, scene_data, use_name)
    dump_list.append(output_skill)



print(len(dump_list))

with open("../scene_sampling/sam_valid_unseen_20_skills.json", "w") as f:
    json.dump(dump_list, f, indent=4)




























# trial_7_8_path = "../sam_length_7_8_trials.json"
# with open(trial_7_8_path, "r") as f:
#     trial_7_8 = json.load(f)

# num = 0
# traj_json_path = "../data/json_2.1.0_merge_goto/train/" 


# dump_list = []

# for trial in trial_7_8:
#     traj_file = os.path.join(traj_json_path, trial, "augmented_traj_data_new.json")
#     with open(traj_file, "r") as f:
#         traj_data = json.load(f)

#     annotations = traj_data["turk_annotations"]
#     num_idx = len(annotations)
#     repeat_id = random.randint(0, num_idx - 1)

#     step_annotations = annotations["anns"][repeat_id]["high_descs"]
#     goal_annotations = annotations["anns"][repeat_id]["task_desc"]

#     sample_ids = list(np.arange(0, len(step_annotations)))
#     sample_ids = [int(i) for i in sample_ids]
#     output_skill = generate_skill(step_annotations, sample_ids, traj_data, repeat_id, scene_data, trial)
#     output_skill["annotation"] = goal_annotations

#     dump_list.append(output_skill)


#     num += 1
#     print(num)

# with open("train_7_8_whole_traj_chained.json", "w") as f:
#     json.dump(dump_list, f, indent=4)
    

