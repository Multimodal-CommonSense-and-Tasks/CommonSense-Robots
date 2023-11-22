import os
import json
import numpy as np
import random
import json
from collections import defaultdict, Counter

#scene_type = "valid_unseen"
scene_type = "valid_seen"

#ann_human_file_name_generator = (
#    lambda scene_type: f"../scene_sampling/bootstrap_{scene_type}_ann_human.json"
#)
ann_human_file_name_generator = (
    lambda scene_type: f"../scene_sampling/bootstrap_{scene_type}-40_ann_human.json"
)


def process_skill_strings(strings):
    processed_strings = []
    for string in strings:
        string = string.strip().lower()
        if string[-1] not in ["?", ".", "!"]:
            string = string + "."
        processed_strings.append(string.capitalize())
    return processed_strings


# generate eval_skills for a specific scene and seed
NUM_FLOOR_PLANS_TO_SAMPLE = 5
NUM_TASKS_PER_FLOOR_PLAN = 5
#lengths = []
#LENGTH_TYPES = [2, 3, 4]
#NUM_TASKS_PER_LENGTH_TYPE = NUM_TASKS_PER_FLOOR_PLAN // len(LENGTH_TYPES)


def load_stuff(scene_type):
    path = f"../data/json_2.1.0_merge_goto/{scene_type}"
    scene_path = f"../scene_sampling/{scene_type}_scene.json"
    splits_path = "../data/splits/oct21.json"
    with open(splits_path) as f:
        splits = json.load(f)
    trials = splits[scene_type]
    with open(scene_path, "r") as f:
        scene_data = json.load(f)
    scene_numbers = [scene_data["scene_num"] for scene_data in scene_data]
    #scene_random_seeds = [scene["random_seed"] for scene in scene_data]
    trial_names = set([scene["task"].split("/")[1] for scene in scene_data])
    # print(scene_numbers, scene_random_seeds)
    # filter for trials with floor plans in our scene_data and load their traj_datas
    # first filter for floor plan
    filtered_trials = [
        trial
        for trial in trials
        if int(trial["task"].split("/")[0].split("-")[-1]) in scene_numbers
    ]
    #final_filtered_trials = []
    trial_keys_to_keep = [
        "task_id",
        "task_type",
        "turk_annotations",
        "plan",
        "scene",
    ]
    floor_plans_so_far = {}
    # to make sure there are equal number of types for each length
    unique_task_ids = set()
    # check if we have enough tasks for each floor plan
    floor_plan_checker = lambda x: sum([len(x[floor_plan]) >= NUM_TASKS_PER_FLOOR_PLAN for floor_plan in x]) < NUM_FLOOR_PLANS_TO_SAMPLE
    # sample NUM_FLOOR_PLANS_TO_SAMPLE floor plans
    while len(floor_plans_so_far) < NUM_FLOOR_PLANS_TO_SAMPLE or floor_plan_checker(floor_plans_so_far):
        filtered_trial = random.choice(filtered_trials)
        file_name = filtered_trial["task"]
        file_path = os.path.join(path, file_name, "augmented_traj_data_new.json")
        with open(file_path, "r") as f:
            traj_data = json.load(f)
        if traj_data["task_id"] in trial_names and traj_data["task_id"] not in unique_task_ids:
            unique_task_ids.add(traj_data["task_id"])
            if traj_data["scene"]["scene_num"] not in floor_plans_so_far:
                floor_plans_so_far[traj_data["scene"]["scene_num"]] = []
            length_of_task = len(traj_data['plan']['high_pddl']) - 1
            #if length_of_task > max(LENGTH_TYPES): # consider all length 5 tasks the same
            #    length_of_task = max(LENGTH_TYPES)
            if length_of_task < 2:
                continue
            #if length_of_task < 1:
            #    continue
            if len(floor_plans_so_far[traj_data["scene"]["scene_num"]]) < NUM_TASKS_PER_FLOOR_PLAN:# 
                #lengths.append(length_of_task)
                floor_plans_so_far[traj_data["scene"]["scene_num"]].append(file_name)
                print({floor_plan: len(floor_plans_so_far[floor_plan]) for floor_plan in floor_plans_so_far})

    # now remove excess floor-plans from floor_plans_so_far that have less than NUM_TASKS_PER_FLOOR_PLAN
    new_floor_plans_so_far = {}
    for floor_plan in floor_plans_so_far:
        if len(floor_plans_so_far[floor_plan]) >= NUM_TASKS_PER_FLOOR_PLAN:
            new_floor_plans_so_far[floor_plan] = floor_plans_so_far[floor_plan]
    floor_plans_so_far = new_floor_plans_so_far
    # if there are still more than NUM_FLOOR_PLANS_TO_SAMPLE floor plans, remove the excess
    if len(floor_plans_so_far) > NUM_FLOOR_PLANS_TO_SAMPLE:
        floor_plans_so_far = {floor_plan: floor_plans_so_far[floor_plan] for floor_plan in random.sample(list(floor_plans_so_far), NUM_FLOOR_PLANS_TO_SAMPLE)}
    assert len(floor_plans_so_far) == NUM_FLOOR_PLANS_TO_SAMPLE
    final_filtered_trials = []
    for floor_plan in floor_plans_so_far:
        for file_name in floor_plans_so_far[floor_plan]:
            filtered_traj_data = {}
            file_path = os.path.join(path, file_name, "augmented_traj_data_new.json")
            with open(file_path, "r") as f:
                traj_data = json.load(f)
            for key in trial_keys_to_keep:
                filtered_traj_data[key] = traj_data[key]
            filtered_traj_data["task"] = file_name
            final_filtered_trials.append(filtered_traj_data)
    #final_filtered_trials = [trial for trial in final_filtered_trials if trial["task"] in file_names_to_keep_in_final_filtered_trials]
    assert len(final_filtered_trials) == NUM_FLOOR_PLANS_TO_SAMPLE * NUM_TASKS_PER_FLOOR_PLAN
    # get the lengths again
    lengths = []
    for trial in final_filtered_trials:
        length_of_task = len(trial['plan']['high_pddl']) - 1
        lengths.append(length_of_task)
    print(np.mean(lengths), Counter(lengths))
    return final_filtered_trials, scene_data


def grab_skills_at_index(final_filtered_trials, scene_data, i):
    # select trajectory
    traj_data = final_filtered_trials[i]
    # randomly select repeat_id
    repeat_ids = list(range(len(traj_data["turk_annotations"]["anns"])))
    generated_eval_skills = []
    #for repeat_id in repeat_ids:
    repeat_id = random.choice(repeat_ids)
    annotations = traj_data["turk_annotations"]["anns"]
    annotations = annotations[repeat_id]["high_descs"]
    sample_ids = list(range(len(annotations)))
    generated_eval_skill = generate_skill(
        annotations,
        sample_ids,
        traj_data,
        repeat_id,
        scene_data,
    )
    generated_eval_skill["annotation"] = process_skill_strings(
        [traj_data["turk_annotations"]["anns"][repeat_id]["task_desc"]]
    )[0]
    generated_eval_skills.append(generated_eval_skill)
    return generated_eval_skills


def grab_consecutive_ann(traj_data, repeat_id):
    annotations = traj_data["turk_annotations"]["anns"]
    annotations = annotations[repeat_id]["high_descs"]

    while True:
        sample_ids = random.sample(range(0, (len(annotations) + 1)), 2)
        if np.abs(sample_ids[0] - sample_ids[1]) > 1:
            break
    sample_ids = random.sample(range(0, (len(annotations) + 1)), 2)
    sample_ids.sort()
    annotations = annotations[sample_ids[0] : sample_ids[1]]
    sample_ids = list(range(sample_ids[0], sample_ids[1]))

    return annotations, sample_ids


def generate_skill(annotations, sample_ids, traj_data, repeat_id, scene_data):
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
        skill["annotations"] = annotations[skill_index]

    generated_skill["task"] = file_name
    generated_skill["repeat_id"] = repeat_id
    generated_skill["subgoal_ids"] = list(sample_ids)
    generated_skill["primitive_skills"] = matching_skills
    return generated_skill


def save_eval_dataset(new_eval_data, file_name):
    total_eval_data = new_eval_data
    with open(file_name, "w") as f:
        json.dump(total_eval_data, f, indent=4)


if __name__ == "__main__":
    (
        final_filtered_trials,
        scene_data,
    ) = load_stuff(scene_type)
    eval_skills = []
    for i in range(len(final_filtered_trials)):
        skills_of_interest = grab_skills_at_index(final_filtered_trials, scene_data, i)
        eval_skills.extend(skills_of_interest)
    ann_human_file_name = ann_human_file_name_generator(scene_type)
    save_eval_dataset(eval_skills, ann_human_file_name)

