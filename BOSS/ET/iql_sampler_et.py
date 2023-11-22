import numpy as np
import revtok
import torch
import os
import sys

from env.thor_env import ThorEnv

path = "."
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(path, "gen"))
sys.path.append(os.path.join(path, "models"))
sys.path.append(os.path.join(path, "models", "eval"))
import json
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from ET.et_iql_feeder import remove_spaces_and_lower, numericalize
from eval_utils import (
    process_skill_strings,
    forward_model_object_id_ability_choice_iql,
    knives,
)
from utils import AttrDict, load_object_class
from gen.constants import VISIBILITY_DISTANCE
import random
from eval_utils import (
    compute_distance,
    compute_visibility_based_on_distance,
    interactive_actions,
    mask_and_resample,
)

vocab = torch.load("pp.vocab")
vocab_obj = torch.load("data/obj_cls.vocab")

hard_sample = [
    {
        "task": "pick_heat_then_place_in_recep-AppleSliced-None-CounterTop-14/trial_T20190907_232225_725376",
        "repeat_id": 1,
        "subgoal_id": 4,  # 6,
        "subgoal_type": "HeatObject",
        "return": "7.6",
        "ann": "Put the piece of apple in the microwave and cook it for a few seconds, before taking it back out",
    },
    # {
    #    "task": "pick_heat_then_place_in_recep-AppleSliced-None-CounterTop-14/trial_T20190907_232225_725376",
    #    "repeat_id": 1,
    #    "subgoal_id": 7,
    #    "subgoal_type": "GotoLocation",
    #    "return": "3.0",
    #    "ann": "Walk to your left and face the counter top",
    # },
    {
        "task": "pick_heat_then_place_in_recep-AppleSliced-None-CounterTop-14/trial_T20190907_232225_725376",
        "repeat_id": 1,
        "subgoal_id": 5,  # 8,
        "subgoal_type": "PutObject",
        "return": "2",
        "ann": "Set the piece of apple down on the counter top",
    },
    "Put the piece of apple in the microwave and cook it for a few seconds, before taking it back out. Set the piece of apple down on the counter top.",
]

medium_sample = [
    # {
    #    "task": "pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-13/trial_T20190909_115736_122556",
    #    "repeat_id": 2,
    #    "subgoal_id": 4,
    #    "subgoal_type": "GotoLocation",
    #    "return": "5.0",
    #    "ann": "Turn right, turn right, walk straight to the oven",
    # },
    {
        "task": "pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-13/trial_T20190909_115736_122556",
        "repeat_id": 2,
        "subgoal_id": 2,  # 5,
        "subgoal_type": "PutObject",
        "return": "1.9",
        "ann": "Open the microwave, put the knife inside, close the microwave",
    },
    "Walk to the oven and put the knife inside the microwave.",
]

easy_sample = [
    # {
    #    "task": "pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-13/trial_T20190909_115736_122556",
    #    "repeat_id": 0,
    #    "subgoal_id": 6,
    #    "subgoal_type": "GotoLocation",
    #    "return": "4.8",
    #    "ann": "turn to the left twice and take a few step and turn to the right and go to the refrigerator",
    # },
    {
        "task": "pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-13/trial_T20190909_115736_122556",
        "repeat_id": 0,
        "subgoal_id": 3,  # 7,
        "subgoal_type": "PickupObject",
        "return": "1.9",
        "ann": "open the refrigerator door and take a potato slice from the shelf and close the refrigerator door",
    },
    "Take a potato slice from the refrigerator.",
]

subgoal_pool = [easy_sample, medium_sample, hard_sample]

MAX_FAILS = 1
MAX_STEPS = 30 * 3
DATA_PATH = "data/json_2.1.0_merge_goto/preprocess"
VISUAL_MODEL = "resnet18"
REWARD_CONFIG = "models/config/rewards.json"

global_task_args = AttrDict()
global_task_args.reward_config = REWARD_CONFIG
global_task_args.visual_model = VISUAL_MODEL
global_task_args.max_steps = MAX_STEPS


def process_annotation(annotation, model):
    ann_l = revtok.tokenize(remove_spaces_and_lower(annotation))
    ann_l = [w.strip().lower() for w in ann_l]
    ann_token = numericalize(model.vocab_word, ann_l, train=False)
    ann_token = torch.tensor(ann_token).long()
    return ann_token


def setup_scene(env: ThorEnv, traj_data, r_idx, args):
    """
    intialize the scene and agent from the task info
    """
    # scene setup
    scene_num = traj_data["scene"]["scene_num"]
    object_poses = traj_data["scene"]["object_poses"]
    dirty_and_empty = traj_data["scene"]["dirty_and_empty"]
    object_toggles = traj_data["scene"]["object_toggles"]

    scene_name = "FloorPlan%d" % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    env.step(dict(traj_data["scene"]["init_action"]))
    # print goal instr
    # print("Task: %s" % (traj_data["turk_annotations"]["anns"][r_idx]["task_desc"]))

    # setup task for reward
    env.set_task(traj_data, args, reward_type="dense")


def sample_task(
    eval_split,
    success,
    model_specific_json,
    num_subgoals_in_pool,
    selected_specific_subgoal,
    which_task,
):
    # get from saved jsons for this file which to sample
    # num_subgoals_in_pool tells us what the max size of the subgoal pool to sample from is
    # if num_subgoals_inPool = -1, then we sample from the entire pool
    if (
        num_subgoals_in_pool > 1 and selected_specific_subgoal is None
    ) or num_subgoals_in_pool == 1:
        selected_specific_subgoal = random.randint(0, num_subgoals_in_pool - 1)
    logs = subgoal_pool[selected_specific_subgoal]
    log = logs[0]

    task = log["task"]
    REPEAT_ID = log["repeat_id"]
    eval_idx = log["subgoal_id"]
    json_path = os.path.join(DATA_PATH, eval_split, task, "ann_%d.json" % REPEAT_ID)
    with open(json_path) as f:
        traj_data = json.load(f)
    return eval_idx, traj_data, REPEAT_ID, selected_specific_subgoal


def run_policy(
    env,
    model,
    visual_preprocessor,
    model_specific_json,
    device,
    eval_split,
    traj_type,
    max_skill_length,
    num_subgoals_in_pool,
    deterministic,
    use_fractional_rewards,
    use_primitive_skill_embedding,
    skill_embedding_map,
    primitive_to_combined_embedding_switch_point,
    log_video,
    epsilon=0.1,
    selected_specific_subgoal=None,
):
    model.eval()
    with torch.no_grad():
        eval_idx, traj_data, r_idx, actually_selected_subgoal = sample_task(
            eval_split,
            traj_type,
            model_specific_json,
            num_subgoals_in_pool,
            selected_specific_subgoal,
            0,
        )
        num_subgoals_to_complete = len(subgoal_pool[actually_selected_subgoal]) - 1

        setup_scene(env, traj_data, r_idx, global_task_args)

        expert_init_actions = [
            a["discrete_action"]
            for a in traj_data["plan"]["low_actions"]
            if a["high_idx"] < eval_idx
        ]

        all_instructions = process_skill_strings(
            [
                subgoal_pool[actually_selected_subgoal][i]["ann"]
                for i in range(len(subgoal_pool[actually_selected_subgoal]) - 1)
            ]
            + [subgoal_pool[actually_selected_subgoal][-1]]
        )
        # subgoal info
        chained_subgoal_instr = all_instructions[-1]
        first_primitive_subgoal_instr = all_instructions[0]
        if chained_subgoal_instr not in skill_embedding_map:
            num_attempts = 0
            print(f"{chained_subgoal_instr} not in skill embedding map")
        else:
            num_attempts = len(skill_embedding_map[chained_subgoal_instr])

        # if not enough attempts, or we're trying to use the primitive skill embeddings and we're not successful at chaining these yet
        if (use_primitive_skill_embedding and num_attempts < 2) or (
            use_primitive_skill_embedding
            and (
                np.mean(skill_embedding_map[chained_subgoal_instr])
                < primitive_to_combined_embedding_switch_point
            )
        ):
            using_primitive_skills_this_rollout = True
            first_language_ann = first_primitive_subgoal_instr
        # otherwise enough attempts and we're using primitive skill embeddings, but we're successful enough, we can try chaining
        elif (
            num_attempts >= 2
            and use_primitive_skill_embedding
            and np.mean(skill_embedding_map[chained_subgoal_instr])
            >= primitive_to_combined_embedding_switch_point
        ):
            using_primitive_skills_this_rollout = False
            first_language_ann = chained_subgoal_instr
        # in all other cases anyway we'll be using chained language latent
        else:
            using_primitive_skills_this_rollout = False
            first_language_ann = chained_subgoal_instr
        feat = {}
        feat["language_ann"] = (
            process_annotation(first_language_ann, model).to(device).unsqueeze(0)
        )
        primitive_language_ann = process_annotation(
            first_primitive_subgoal_instr, model
        )
        per_step_lang_anns = []
        completed_eval_idx = eval_idx + num_subgoals_to_complete - 1
        start_idx = eval_idx
        done = 0
        t = 0
        fails = 0
        switch = 0

        obs = []
        acs = []
        obj_acs = []
        dones = []
        env_rewards = []
        str_act = []
        skill_switch_points = []
        while not done:
            # break if max_steps reached
            if t >= MAX_STEPS + len(expert_init_actions):
                break

            if (len(expert_init_actions) == 0 and t == 0) or (len(expert_init_actions)!= 0 and t == len(expert_init_actions)):
                curr_image = Image.fromarray(np.uint8(env.last_event.frame))

                curr_frame = visual_preprocessor.featurize([curr_image], batch=1).to(
                    device
                )
                feat["frames_buffer"] = curr_frame.unsqueeze(0).to(device)
                feat["action_traj"] = torch.zeros(1, 0).long().to(device)
                feat["object_traj"] = torch.zeros(1, 0).long().to(device)

            if t < len(expert_init_actions):

                # get expert action

                action = expert_init_actions[t]
                # print("expert_action", action)
                compressed_mask = (
                    action["args"]["mask"] if "mask" in action["args"] else None
                )
                mask = (
                    env.decompress_mask(compressed_mask)
                    if compressed_mask is not None
                    else None
                )

                success, _, _, err, _ = env.va_interact(
                    action["action"], interact_mask=mask, smooth_nav=True, debug=False
                )
                if not success:
                    # print("expert initialization failed, re-sampling")
                    return run_policy(
                        env,
                        model,
                        visual_preprocessor,
                        model_specific_json,
                        device,
                        eval_split,
                        traj_type,
                        max_skill_length,
                        num_subgoals_in_pool,
                        deterministic,
                        use_fractional_rewards,
                        use_primitive_skill_embedding,
                        skill_embedding_map,
                        primitive_to_combined_embedding_switch_point,
                        log_video,
                        epsilon,
                        selected_specific_subgoal=selected_specific_subgoal,
                    )
                _, _ = env.get_transition_reward()
            else:
                obs.append(curr_frame.cpu().detach())
                per_step_lang_anns.append(primitive_language_ann.cpu())
                skill_switch_points.append(switch)
                if (len(expert_init_actions) == 0 and t == 0) or (len(expert_init_actions)!= 0 and t == len(expert_init_actions)):
                    video_frames = []
                    value_predict = []
                video_frames.append(np.uint8(env.last_event.frame))
                (action, output_object, value_output) = get_action_from_agent(
                    model,
                    feat,
                    vocab,
                    vocab_obj,
                    env,
                    deterministic=deterministic,
                    epsilon=epsilon,
                    ret_value=True,
                )
                if value_output != None:
                    value_output = value_output.squeeze().item()
                    value_predict.append(value_output)
                try:
                    _, _ = env.to_thor_api_exec(action, output_object, smooth_nav=True)
                except Exception as e:
                    # if there's an exception from running, then we'll just try again.
                    # print(e)
                    # record this exception in exceptions.txt
                    with open("exceptions.txt", "a") as f:
                        f.write(str(e) + "\n")

                next_frame = np.uint8(env.last_event.frame)
                next_frame = Image.fromarray(next_frame)
                next_frame = visual_preprocessor.featurize([next_frame], batch=1).to(
                    device
                )
                acs.append(vocab["action_low"].word2index(action) - 2)
                obj_index = load_object_class(vocab_obj, output_object)
                str_act.append(
                    dict(
                        action=action,
                        object=output_object.split("|")[0]
                        if output_object is not None
                        else None,
                    )
                )
                obj_acs.append(obj_index)
                curr_frame = next_frame
                feat["frames_buffer"] = torch.cat(
                    [feat["frames_buffer"], next_frame.unsqueeze(0)], dim=1
                ).to(device)
                # - 2 instead of -3 like for regular rollouts because ET dataloader had a -1 instead of -2 for padding reasons
                tensor_action = torch.tensor(
                    vocab["action_low"].word2index(action) - 2
                ).to(device)
                feat["action_traj"] = torch.cat(
                    [feat["action_traj"], tensor_action.unsqueeze(0).unsqueeze(0)],
                    dim=1,
                )
                feat["object_traj"] = torch.cat(
                    [
                        feat["object_traj"],
                        torch.tensor(obj_index).unsqueeze(0).unsqueeze(0).to(device),
                    ],
                    dim=1,
                )

                feat["frames_buffer"] = feat["frames_buffer"][:, -max_skill_length:]
                feat["action_traj"] = feat["action_traj"][:, -max_skill_length + 1 :]
                feat["object_traj"] = feat["object_traj"][:, -max_skill_length + 1 :]

                t_success = env.last_event.metadata["lastActionSuccess"]
                if not t_success:
                    err = env.last_event.metadata["errorMessage"]
                    fails += 1
                    if fails >= MAX_FAILS:
                        exception_string = f"Failed to execute action {action} after {MAX_FAILS} tries. Object: {output_object}. Error: {err}"
                        with open("exceptions.txt", "a") as f:
                            f.write(exception_string + "\n")
                        # print(exception_string)

                # MUST call get_transition_reward to update the environment
                t_reward, _ = env.get_transition_reward()
                curr_subgoal_idx = env.get_subgoal_idx()
                # print("curr_subgoal_idx",curr_subgoal_idx)
                partial_success = 0
                if curr_subgoal_idx == completed_eval_idx:
                    done = 1
                    partial_success = 1
                elif curr_subgoal_idx == eval_idx:
                    eval_idx += 1
                    # if using primitive skills, give next subgoal's instruction as language latent
                    subgoal_instr = all_instructions[eval_idx - start_idx]
                    primitive_language_ann = process_annotation(subgoal_instr, model)
                    if using_primitive_skills_this_rollout:
                        # if using primitive skills and from_empty init, re-init to make it like the primitive skills
                        feat["frames_buffer"] = next_frame.unsqueeze(0).to(device)
                        feat["action_traj"] = torch.zeros(1, 0).long().to(device)
                        feat["object_traj"] = torch.zeros(1, 0).long().to(device)
                        feat["language_ann"] = primitive_language_ann.unsqueeze(0).to(
                            device
                        )
                    switch = 1
                    partial_success = 1
                else:
                    switch = 0
                if use_fractional_rewards:
                    env_rewards.append(
                        partial_success * 1 / (len(subgoal_pool) - 1)
                    )  # fractional rewards
                else:
                    env_rewards.append(partial_success)

            t = t + 1
        subgoal_last_frame_video = np.uint8(env.last_event.frame)
        (*_, value_output) = get_action_from_agent(
            model,
            feat,
            vocab,
            vocab_obj,
            env,
            deterministic=deterministic,
            epsilon=epsilon,
            ret_value=True,
        )
        video_frames.append(subgoal_last_frame_video)
        obs.append(next_frame.cpu().detach())  # next obs

        if value_output != None:
            value_output = value_output.squeeze().item()
            value_predict.append(value_output)

        if log_video:
            value_font = ImageFont.truetype("FreeMono.ttf", 20)
            action_font = ImageFont.truetype("FreeMono.ttf", 14)
            gif_logs = []
            for frame_number in range(len(video_frames)):
                img = video_frames[frame_number]
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                value_log = value_predict[frame_number]
                if frame_number != len(video_frames) - 1:
                    action_log, object_log = (
                        str_act[frame_number]["action"],
                        str_act[frame_number]["object"],
                    )
                    draw.text(
                        (1, 1),
                        f"Action: {action_log}\nObject: {str(object_log)}",
                        fill=(255, 255, 255),
                        font=action_font,
                    )

                draw.text(
                    (1, 280),
                    "Value: %.3f" % (value_log),
                    fill=(255, 255, 255),
                    font=value_font,
                )
                log_images = np.array(img)
                gif_logs.append(log_images)

            video_frames = np.asarray(gif_logs)
            video_frames = np.transpose(video_frames, (0, 3, 1, 2))

        rewards = torch.tensor(env_rewards, dtype=torch.float)
        dones = torch.zeros(len(rewards))
        dones[-1] = done
        vid_caption = f"{chained_subgoal_instr}: {'SUCCESS' if done else 'FAIL'}. Primitive skills: {using_primitive_skills_this_rollout}"
        return dict(
            obs=torch.cat(obs),
            acs=torch.tensor(acs),
            obj_acs=torch.tensor(obj_acs),
            rews=rewards,
            dones=dones,
            chained_language_instruction=process_annotation(
                chained_subgoal_instr, model
            ),
            per_step_language_instruction=per_step_lang_anns,
            skill_switch_points=torch.tensor(skill_switch_points),
            video_frames=video_frames if log_video else None,
            video_caption=vid_caption,
            subgoal_instr=chained_subgoal_instr,
            first_primitive_instr=all_instructions[0],
            second_primitive_instr=all_instructions[1],
        )


def run_policy_multi_process(
    ret_queue, task_queue, offline_rl_model, resnet, model_specific_json, device,
):
    env = ThorEnv()
    while True:
        task_args = task_queue.get()
        if task_args is None:
            break
        ret_queue.put(
            run_policy(
                env, offline_rl_model, resnet, model_specific_json, device, *task_args,
            )
        )
    env.stop()


def generate_invalid_action_mask_and_objects(env, visible_object, vocab_obj, vocab):
    # we will first filter out all the interact actions that are not available
    def filter_objects(condition):
        condition_input = condition
        if condition == "toggleon" or condition == "toggleoff":
            condition = "toggleable"

        if condition == "openable" or condition == "closeable":
            condition = "openable"

        visible_candidate_objects = [
            obj for obj in visible_object if obj[condition] == True
        ]

        candidate_obj_type = [
            vis_obj["objectId"].split("|")[0] for vis_obj in visible_candidate_objects
        ]

        remove_indicies = []

        if condition_input == "toggleon":
            # print(visible_object)
            if "Faucet" in candidate_obj_type:
                # SinkBasin: Sink|+03.08|+00.89|+00.09|SinkBasin
                visible_object_name = [
                    obj["objectId"].split("|")[-1] for obj in visible_object
                ]
                if "SinkBasin" not in visible_object_name:
                    remove_indicies.append(candidate_obj_type.index("Faucet"))

            for i, obj in enumerate(visible_candidate_objects):
                if (
                    obj["isToggled"] == True
                    and obj["objectId"].split("|")[0] in candidate_obj_type
                ):
                    remove_indicies.append(i)
            # if "CellPhone" in candidate_obj_type:
            #     candidate_obj_type.remove("CellPhone")

        elif condition_input == "toggleoff":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isToggled"] == False:
                    remove_indicies.append(i)
            # if "CellPhone" in candidate_obj_type:
            #     candidate_obj_type.remove("CellPhone")

        elif condition_input == "openable":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isOpen"] == True or obj["isToggled"] == True:
                    remove_indicies.append(i)
                # print(candidate_obj_type)

        elif condition_input == "closeable":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isOpen"] == False:
                    remove_indicies.append(i)

        elif condition_input == "receptacle":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["openable"] == True and obj["isOpen"] == False:
                    remove_indicies.append(i)

        elif condition_input == "sliceable":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isSliced"] == True:
                    remove_indicies.append(i)

        remove_indicies = set(remove_indicies)
        filtered_candidate_obj_type = [
            j for i, j in enumerate(candidate_obj_type) if i not in remove_indicies
        ]
        filtered_visible_candidate_objects = [
            j
            for i, j in enumerate(visible_candidate_objects)
            if i not in remove_indicies
        ]

        candidate_obj_type_id = [
            vocab_obj.word2index(candidate_obj_type_use)
            for candidate_obj_type_use in filtered_candidate_obj_type
            if candidate_obj_type_use in vocab_obj.to_dict()["index2word"]
        ]
        candidate_obj_type_id = np.array(list(set(candidate_obj_type_id)))
        return filtered_visible_candidate_objects, candidate_obj_type_id

    pickupable_object_names, pickupable_objects = filter_objects("pickupable")
    openable_object_names, openable_objects = filter_objects("openable")
    sliceable_object_names, sliceable_objects = filter_objects("sliceable")
    closeable_object_names, closeable_objects = filter_objects("closeable")
    receptacle_object_names, receptacle_objects = filter_objects("receptacle")

    # print("receptacle_object_names",receptacle_object_names)
    toggleon_object_names, toggleon_objects = filter_objects("toggleon")
    toggleoff_object_names, toggleoff_objects = filter_objects("toggleoff")

    # toggleable_object_names, toggleable_objects = filter_objects("toggleable")

    # generate invalid mask
    invalid_action_mask = []
    if (
        len(pickupable_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) > 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("PickupObject") - 2)
    if len(openable_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("OpenObject") - 2)
    if (
        len(sliceable_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) == 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("SliceObject") - 2)
    if len(closeable_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("CloseObject") - 2)
    if (
        len(receptacle_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) == 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("PutObject") - 2)
    if len(toggleon_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("ToggleObjectOn") - 2)
    if len(toggleoff_objects) == 0:
        invalid_action_mask.append(
            vocab["action_low"].word2index("ToggleObjectOff") - 2
        )
    if (
        len(env.last_event.metadata["inventoryObjects"]) > 0
        and env.last_event.metadata["inventoryObjects"][0]["objectId"].split("|")[0]
        not in knives
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("SliceObject") - 2)

    # <<stop>> action needs to be invalid
    invalid_action_mask.append(vocab["action_low"].word2index("<<stop>>") - 2)
    invalid_action_mask = list(set(invalid_action_mask))

    ret_dict = dict(
        pickupable_object_names=pickupable_object_names,
        pickupable_objects=pickupable_objects,
        openable_object_names=openable_object_names,
        openable_objects=openable_objects,
        sliceable_object_names=sliceable_object_names,
        sliceable_objects=sliceable_objects,
        closeable_object_names=closeable_object_names,
        closeable_objects=closeable_objects,
        receptacle_object_names=receptacle_object_names,
        receptacle_objects=receptacle_objects,
        toggleon_object_names=toggleon_object_names,
        toggleon_objects=toggleon_objects,
        toggleoff_object_names=toggleoff_object_names,
        toggleoff_objects=toggleoff_objects,
    )
    return invalid_action_mask, ret_dict


def get_action_from_agent(
    model, feat, vocab, vocab_obj, env, deterministic, epsilon, ret_value,
):
    take_rand_action = random.random() < epsilon

    action_out, object_pred_id, value = model.step(feat, ret_value=ret_value)

    action_out = torch.softmax(action_out, dim=1)

    object_pred_prob = torch.softmax(object_pred_id, dim=1)

    agent_position = env.last_event.metadata["agent"]["position"]

    visible_object = [
        obj
        for obj in env.last_event.metadata["objects"]
        if (
            obj["visible"] == True
            and compute_visibility_based_on_distance(
                agent_position, obj, VISIBILITY_DISTANCE
            )
        )
    ]
    invalid_action_mask, ret_dict = generate_invalid_action_mask_and_objects(
        env, visible_object, vocab_obj, vocab
    )
    # choose the action after filtering with the mask
    chosen_action = mask_and_resample(
        action_out, invalid_action_mask, deterministic, take_rand_action
    )
    string_act = vocab["action_low"].index2word(chosen_action + 2)
    assert string_act != "<<stop>>"
    if string_act not in interactive_actions:
        return string_act, None, value
    object_pred_prob = object_pred_prob.squeeze(0).cpu().detach().numpy()
    # otherwise, we need to choose the closest visible object for our action
    string_act_to_object_list_map = dict(
        PickupObject=(
            ret_dict["pickupable_object_names"],
            ret_dict["pickupable_objects"],
        ),
        OpenObject=(ret_dict["openable_object_names"], ret_dict["openable_objects"]),
        SliceObject=(ret_dict["sliceable_object_names"], ret_dict["sliceable_objects"]),
        CloseObject=(ret_dict["closeable_object_names"], ret_dict["closeable_objects"]),
        PutObject=(ret_dict["receptacle_object_names"], ret_dict["receptacle_objects"]),
        ToggleObjectOn=(
            ret_dict["toggleon_object_names"],
            ret_dict["toggleon_objects"],
        ),
        ToggleObjectOff=(
            ret_dict["toggleoff_object_names"],
            ret_dict["toggleoff_objects"],
        ),
    )

    candidate_object_names, candidate_object_ids = string_act_to_object_list_map[
        string_act
    ]
    prob_dict = {}
    for id in candidate_object_ids:
        if take_rand_action:
            prob_dict[id] = 1
        else:
            prob_dict[id] = object_pred_prob[id]
    prob_value = prob_dict.values()
    if deterministic:
        max_prob = max(prob_value)
        choose_id = [k for k, v in prob_dict.items() if v == max_prob][0]
    else:
        # sample from the object prob distribution
        object_probs = torch.tensor(list(prob_value), dtype=torch.float32)
        if torch.all(object_probs == 0):
            object_probs = torch.ones_like(object_probs)
        choose_id = torch.multinomial(object_probs, 1)[0].item()
        choose_id = list(prob_dict.keys())[choose_id]

    # choose the closest object
    object_type = vocab_obj.index2word(choose_id)
    candidate_objects = [
        obj
        for obj in candidate_object_names
        if obj["objectId"].split("|")[0] == object_type
    ]
    # object type
    agent_position = env.last_event.metadata["agent"]["position"]
    min_distance = float("inf")
    for ava_object in candidate_objects:
        obj_agent_dist = compute_distance(agent_position, ava_object)
        if obj_agent_dist < min_distance:
            min_distance = obj_agent_dist
            output_object = ava_object["objectId"]
    return string_act, output_object, value
