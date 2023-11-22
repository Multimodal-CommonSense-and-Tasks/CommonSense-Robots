import numpy as np
import torch
import os
import sys
import json
import pprint

splits_path = "../data/splits/oct21.json"
data_path = "../data/json_2.1.0_merge_goto/preprocess"
eval_split = "valid_unseen"


with open(splits_path) as f:
    splits = json.load(f)
    pprint.pprint({k: len(v) for k, v in splits.items()})

train_scene = splits[eval_split]
train_scene = train_scene[:]
scene_list = []
high_level_list = []
problem_task = []


for i in range(len(train_scene)):
    # print(i)
    task = train_scene[i]

    json_path = os.path.join(
        data_path, eval_split, task["task"], "ann_%d.json" % task["repeat_idx"],
    )

    with open(json_path) as f:
        traj = json.load(f)

    # scene = traj["scene"]["floor_plan"]
    scene = traj["scene"]
    scene["task"] = task["task"]

    # print(scene)
    print(i, len(scene_list))

    # if (len(traj["plan"]["high_pddl"]) - 1) != len(
    if (len(traj["plan"]["high_pddl"])) != len(
        traj["turk_annotations"]["anns"][task["repeat_idx"]]["high_descs"]
    ):
        problem_task.append(task)

    else:
        if "color_to_object_type" in scene:
            scene.pop("color_to_object_type")
        if len(scene_list) == 0:
            scene_list.append(scene)
        else:
            if scene not in scene_list:
                scene_list.append(scene)

        # for skill_num in range(len(traj["plan"]["high_pddl"]) - 1):
        for skill_num in range(len(traj["plan"]["high_pddl"])):

            skill_dict = {}
            scene_index = scene_list.index(scene)
            skill = traj["plan"]["high_pddl"][skill_num]
            high_idx = skill["high_idx"]
            api_actions = [
                action["discrete_action"]["action"]
                for action in traj["plan"]["low_actions"]
                if action["high_idx"] == high_idx
            ]
            planner_action = skill["planner_action"]
            if "coordinateObjectId" in planner_action:
                planner_action["coordinateObjectId"].pop(1)
            if "coordinateReceptacleObjectId" in planner_action:
                planner_action["coordinateReceptacleObjectId"].pop(1)
            if "forceVisible" in planner_action:
                planner_action.pop("forceVisible")
            discrete_action = skill["discrete_action"]
            annotations = traj["turk_annotations"]["anns"][task["repeat_idx"]][
                "high_descs"
            ][skill_num]

            skill_dict["scene_index"] = scene_index
            skill_dict["planner_action"] = planner_action
            skill_dict["discrete_action"] = discrete_action
            skill_dict["annotations"] = annotations
            skill_dict["api_actions"] = api_actions
            high_level_list.append(skill_dict)


save_scene_path = os.path.join(".", (f"{eval_split}_scene" + ".json"))
save_skill_path = os.path.join(".", (f"{eval_split}_high_skill" + ".json"))
save_error_path = os.path.join(".", ("error_task" + ".json"))

# import random
# random.shuffle(scene_list)
# scene_indicies = [x['scene_num'] for x in scene_list[:5]]
# while len(set(scene_indicies)) < 5:
#    random.shuffle(scene_list)
#    scene_indicies = [x['scene_num'] for x in scene_list[:5]]

with open(save_scene_path, "w") as r:

    json.dump(scene_list, r, indent=4)
# print(len(scene_list))


with open(save_skill_path, "w") as r:

    json.dump(high_level_list, r, indent=4)

with open(save_error_path, "w") as r:

    json.dump(problem_task, r, indent=4)
