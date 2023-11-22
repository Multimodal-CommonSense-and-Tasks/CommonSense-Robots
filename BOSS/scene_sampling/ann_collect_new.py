from asyncio import current_task
from concurrent.futures import process
import os
import copy
import json
import numpy as np
import random
from collections import Counter
import ipywidgets as widgets
from IPython.display import display


def process_skill_strings(strings):
    processed_strings = []
    for string in strings:
        string = string.strip().lower()
        if string[-1] not in ["?", ".", "!"]:
            string = string + "."
        processed_strings.append(string.capitalize())
    return processed_strings


# generate eval_skills for a specific scene and seed
NUM_FLOOR_PLANS_PER_TYPE = 5
LENGTH_TYPES = [1, 2, 3, 4, 5]


def load_stuff(scene_type):
    path = f"../data/json_2.1.0_merge_goto/{scene_type}"
    ann_human_file_name = f"../scene_sampling/{scene_type}_ann_human_roll.json"
    scene_path = f"../scene_sampling/{scene_type}_scene.json"
    splits_path = "../data/splits/oct21.json"

    exist_run = []
    existing_eval_data = []
    if os.path.exists(ann_human_file_name):
        with open(scene_path, "r") as f:
            scene_all = json.load(f)

        with open(ann_human_file_name, "r") as f:
            existing_eval_data = json.load(f)
        final_filtered_trials = []
        for runs in existing_eval_data:
            run = {}
            run["task"] = runs["task"]
            run["subgoal_ids"] = runs["subgoal_ids"]
            run["repeat_id"] = runs["repeat_id"]
            exist_run.append(run)
            final_filtered_trials.append(runs["task"])
        final_filtered_trials = list(set(final_filtered_trials))

        scene_random_seeds = []
        for choosed_trial in final_filtered_trials:
            scene_path = os.path.join(
                path, choosed_trial, "augmented_traj_data_new.json"
            )
            with open(scene_path, "r") as f:
                scene_data = json.load(f)
            scene_random_seeds.append(scene_data["scene"]["random_seed"])
        scene_data = [
            scene for scene in scene_all if scene["random_seed"] in scene_random_seeds
        ]
        # print("scene_data", len(scene_data), "exist_run", len(exist_run))

    else:
        with open(splits_path) as f:
            splits = json.load(f)
        trials = splits[scene_type]
        # all_task = [task["task"] for task in trials]

        with open(scene_path, "r") as f:
            scene_data = json.load(f)
            random.shuffle(scene_data)
            scene_data = scene_data[:NUM_FLOOR_PLANS_PER_TYPE]

        scene_numbers = [scene_data["scene_num"] for scene_data in scene_data]
        scene_random_seeds = [scene["random_seed"] for scene in scene_data]

        # filter for trials with floor plans in our scene_data and load their traj_datas
        # first filter for floor plan
        filtered_trials = [
            trial
            for trial in trials
            if int(trial["task"].split("/")[0].split("-")[-1]) in scene_numbers
        ]

        final_filtered_trials = []
        trial_keys_to_keep = [
            "task_id",
            "task_type",
            "turk_annotations",
            "plan",
            "scene",
        ]
        # then load their trajectories and check the random seed
        for filtered_trial in filtered_trials:
            file_name = filtered_trial["task"]
            file_path = os.path.join(path, file_name, "augmented_traj_data_new.json")
            with open(file_path, "r") as f:
                traj_data = json.load(f)
            if traj_data["scene"]["random_seed"] in scene_random_seeds:
                if scene_random_seeds.index(
                    traj_data["scene"]["random_seed"]
                ) == scene_numbers.index(traj_data["scene"]["scene_num"]):
                    filtered_traj_data = {}
                    for key in trial_keys_to_keep:

                        filtered_traj_data[key] = traj_data[key]
                    filtered_traj_data["task"] = file_name
                    final_filtered_trials.append(file_name)
        final_filtered_trials = list(set(final_filtered_trials))
        # print("Choose scene", len(final_filtered_trials))

        # filtered_trials = list(set([trial["task"] for trial in existing_data]))
    choice_trials = []
    for trial in final_filtered_trials:
        traj_data = json.load(
            open(os.path.join(path, trial, "augmented_traj_data_new.json"), "r")
        )
        num_subgoal = len(traj_data["plan"]["high_pddl"])
        if traj_data["plan"]["high_pddl"][-1]["discrete_action"]["action"] == "NoOp":
            num_subgoal -= 1
        for i in range(num_subgoal):
            subgoal_id = list(np.arange(0, i + 1, 1))
            # print(subgoal_id)
            choice_trials.append({"task": trial, "subgoal_ids": subgoal_id})

    choose_run = []
    for trial in choice_trials:
        temp_dict = trial
        ridx = list(np.arange(3))
        random.shuffle(ridx)
        for i in ridx:
            temp_dict["repeat_id"] = i

            if temp_dict not in exist_run:
                choose_run.append(temp_dict)
                break

    # random.shuffle(choose_run)
    # print(len(choose_run))
    return choose_run, scene_data, existing_eval_data


def sample_from_pool(choose_run, existing_eval_data, scene_data, scene_type):

    while True:
        if len(choose_run) == 0:
            return None
        # runs = len(choose_run) - 1
        # run_index = random.randint(0, runs)
        choose_trial = choose_run.pop(0)
        task = choose_trial["task"]
        subgoal_ids = choose_trial["subgoal_ids"]
        repeat_id = choose_trial["repeat_id"]

        traj_data_new = json.load(
            open(
                os.path.join(
                    f"../data/json_2.1.0_merge_goto/{scene_type}",
                    task,
                    "augmented_traj_data_new.json",
                ),
                "r",
            )
        )
        trial_keys_to_keep = [
            "task_id",
            "task_type",
            "turk_annotations",
            "plan",
            "scene",
        ]
        traj_data = {}
        for key in trial_keys_to_keep:
            traj_data[key] = traj_data_new[key]
        traj_data["task"] = task

        annotations = traj_data["turk_annotations"]["anns"]
        annotations = annotations[repeat_id]["high_descs"]
        annotations = annotations[0 : subgoal_ids[-1] + 1]
        generated_eval_skill = generate_skill(
            annotations, subgoal_ids, traj_data, repeat_id, scene_data
        )

        sampled_length = len(annotations)

        # if the length is 1 then we don't need the human to annotate
        if sampled_length == 1:
            primitive_annotation = annotations[0]
            generated_eval_skill["annotation"] = primitive_annotation
            existing_eval_data.append(generated_eval_skill)
        # otherwise if the length is the full length then we don't need the human to annotate
        elif (
            sampled_length == len(traj_data["plan"]["high_pddl"])
            and traj_data["plan"]["high_pddl"][-1]["discrete_action"]["action"]
            != "NoOp"
        ):
            generated_eval_skill["annotation"] = traj_data["turk_annotations"]["anns"][
                repeat_id
            ]["task_desc"]
            existing_eval_data.append(generated_eval_skill)

        elif (
            sampled_length == len(traj_data["plan"]["high_pddl"]) - 1
            and traj_data["plan"]["high_pddl"][-1]["discrete_action"]["action"]
            == "NoOp"
        ):
            generated_eval_skill["annotation"] = traj_data["turk_annotations"]["anns"][
                repeat_id
            ]["task_desc"]
            existing_eval_data.append(generated_eval_skill)
        else:
            break

    # now, we need to query the human
    widget_text = f"Please write a one-sentence task description that describes the following instructions:\n"
    processed_sampled_annotations = process_skill_strings(annotations)
    for i in range(len(annotations)):
        widget_text += "\n{}. {}".format(i + 1, processed_sampled_annotations[i])
    return widget_text, generated_eval_skill


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
    generated_skill["repeat_id"] = int(repeat_id)
    # print(generated_skill["repeat_id"])

    generated_skill["subgoal_ids"] = [int(idx) for idx in sample_ids]
    generated_skill["primitive_skills"] = matching_skills
    return generated_skill


def save_eval_dataset(existing_eval_data, file_name):
    total_eval_data = existing_eval_data
    with open(file_name, "w") as f:
        json.dump(total_eval_data, f, indent=4)


summary_text = None
skill_of_interest = None
choose_pool = None


def interact_program(annotated_skills, scene_type):
    global skill_of_interest
    global summary_text
    global choose_pool
    global existing_ann

    (choose_pool, scene_data, existing_ann,) = load_stuff(scene_type)

    summary_text, skill_of_interest = sample_from_pool(
        choose_pool, existing_ann, scene_data, scene_type
    )
    prompt = widgets.Textarea(
        value=summary_text,
        description="Skills to summarize:",
        disabled=True,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto", height="300px"),
    )
    text = widgets.Textarea(
        value="",
        description="Write your summary here:",
        rows=1,
        style={"description_width": "initial"},
        layout=widgets.Layout(height="auto", width="auto"),
    )
    error = widgets.Text(
        value="", description="", disabled=True, layout=widgets.Layout(width="auto"),
    )
    submit_button = widgets.Button(
        description="Submit Summary", layout=widgets.Layout(width="auto"),
    )
    # skip_button = widgets.Button(
    #    description="Skip", layout=widgets.Layout(width="auto"),
    # )
    display(error)
    display(prompt)
    display(text)
    # display(skip_button, submit_button)
    display(submit_button)

    def on_submit(b):
        global summary_text
        global skill_of_interest
        if text.value == "":
            error.value = "Please write a summary before submitting."
        else:
            error.value = ""
            formatted_string = process_skill_strings([text.value])
            skill_of_interest["annotation"] = formatted_string
            existing_ann.append(skill_of_interest)
            # print(len(existing_ann))
            rets = sample_from_pool(choose_pool, existing_ann, scene_data, scene_type)
            if rets is None:
                text.value = "You've finished annotating! Thanks!"
                submit_button.disabled = True
                # skip_button.disabled = True
                ann_human_file_name = (
                    f"../scene_sampling/{scene_type}_ann_human_roll.json"
                )
                save_eval_dataset(existing_ann, ann_human_file_name)
            else:
                summary_text, skill_of_interest = rets
                prompt.value = summary_text
                text.value = ""
                submit_button.disabled = False
                # skip_button.disabled = False

    def on_skip(b):
        global summary_text
        global skill_of_interest
        rets = sample_from_pool(choose_pool, existing_ann, scene_data,)
        if rets is None:
            text.value = "You've finished annotating! Thanks!"
            submit_button.disabled = True
            # skip_button.disabled = True
            ann_human_file_name = f"../scene_sampling/{scene_type}_ann_human_roll.json"
            save_eval_dataset(existing_ann, ann_human_file_name)
        else:
            summary_text, skill_of_interest = rets
            prompt.value = summary_text
            text.value = ""
            submit_button.disabled = False
            # skip_button.disabled = False

    submit_button.on_click(on_submit)
    # skip_button.on_click(on_skip)
