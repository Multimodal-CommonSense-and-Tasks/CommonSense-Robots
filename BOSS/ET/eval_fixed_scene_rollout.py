import numpy as np
import torch
import os
import sys
from ET.iql_sampler_et import get_action_from_agent

from env.thor_env import ThorEnv
from utils import generate_video

path = "."
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(path, "gen"))
sys.path.append(os.path.join(path, "models"))
sys.path.append(os.path.join(path, "models", "eval"))
from torch import nn
import json
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from eval_utils import process_skill_strings
from utils import AttrDict, load_object_class

vocab = torch.load("pp.vocab")
vocab_obj = torch.load("data/obj_cls.vocab")


MAX_FAILS = 1
BUFFER_INIT = "from_empty"
DATA_PATH = "data/json_2.1.0_merge_goto/preprocess"
VISUAL_MODEL = "resnet18"
# REWARD_CONFIG = "models/config/rewards.json"
# DEFAULT_NUM_STEPS = 30
EVAL_STEP_RATIO = 5
USE_COMBINED_PRIMITIVE_INSTRUCTIONS = False

global_task_args = AttrDict()
# global_task_args.reward_config = REWARD_CONFIG
global_task_args.visual_model = VISUAL_MODEL
# global_task_args.max_steps = MAX_STEPS
global_task_args.buffer_init = BUFFER_INIT


def generate_skill_info(traj_data, index):
    prev_hl_index = traj_data["plan"]["high_pddl"][index]["high_idx"]
    skill_info_for_online_env = {}
    current_api_actions = [
        action["discrete_action"]["action"]
        for action in traj_data["plan"]["low_actions"]
        if action["high_idx"] == prev_hl_index
    ]
    skill_info_for_online_env["api_actions"] = current_api_actions
    skill_info_for_online_env["planner_action"] = traj_data["plan"]["high_pddl"][index][
        "planner_action"
    ]
    skill_info_for_online_env["discrete_action"] = traj_data["plan"]["high_pddl"][
        index
    ]["discrete_action"]
    skill_info_for_online_env["annotations"] = traj_data["turk_annotations"]["anns"][0][
        "high_descs"
    ][index]
    skill_info_for_online_env["args_object"] = []
    skill_info_for_online_env["args_object"].append("None")
    if skill_info_for_online_env["discrete_action"]["action"] == "GotoLocation":
        skill_info_for_online_env["args_object"].append(
            skill_info_for_online_env["discrete_action"]["args"][0]
        )
    skill_info_for_online_env[
        "scene_index"
    ] = 0  # hardcoded for gotolocation reward calculation

    return skill_info_for_online_env


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
    """ using OLD thor env
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    env.step(dict(traj_data["scene"]["init_action"]))
    # print goal instr
    # print("Task: %s" % (traj_data["turk_annotations"]["anns"][r_idx]["task_desc"]))

    # setup task for reward
    env.set_task(traj_data, args, reward_type="dense")

    old thor env"""
    env.test_reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    env.step(dict(traj_data["scene"]["init_action"]))


def sample_task(scene_type, target_composite_skill_info):
    task = target_composite_skill_info["task"]
    if task[-5:] == ".json":
        # remove the json file from here
        task = "/".join(task.split("/")[:-1])
    REPEAT_ID = target_composite_skill_info["repeat_id"]
    eval_idx = min(target_composite_skill_info["subgoal_ids"])
    json_path = os.path.join(DATA_PATH, scene_type, task, "ann_%d.json" % REPEAT_ID)
    with open(json_path) as f:
        traj_data = json.load(f)
    return eval_idx, traj_data, REPEAT_ID


def eval_policy(
    env,
    model,
    sentence_embedder,
    visual_preprocessor,
    max_skill_length,
    device,
    scene_type,
    num_scenes_to_sample,  # not used here, but kept for consistency
    deterministic,
    log_video,
    composite_skill_list,  # not used here, but kept for consistency
    target_composite_skill_info,
    epsilon=0.1,
    use_combined_primitive_instructions=USE_COMBINED_PRIMITIVE_INSTRUCTIONS,
):
    model.eval()
    with torch.no_grad():
        eval_idx, traj_data, r_idx = sample_task(
            scene_type, target_composite_skill_info,
        )
        num_subgoals_to_complete = len(target_composite_skill_info["primitive_skills"])
        original_eval_idx = eval_idx  # needed because eval_idx will be incremented

        setup_scene(env, traj_data, r_idx, global_task_args)

        expert_init_actions = [
            a["discrete_action"]
            for a in traj_data["plan"]["low_actions"]
            if a["high_idx"] < eval_idx
        ]
        prev_skill_info = generate_skill_info(traj_data, original_eval_idx)

        if use_combined_primitive_instructions:
            composite_instruction = process_skill_strings(
                [
                    skill["annotations"]
                    for skill in target_composite_skill_info["primitive_skills"]
                ]
            )
            composite_instruction = " ".join(composite_instruction)
        else:
            composite_instruction = process_skill_strings(
                [target_composite_skill_info["annotation"]]
            )
        instruction_embedding = sentence_embedder.encode(
            composite_instruction, convert_to_tensor=True
        )
        feat = {}
        feat["language_ann"] = instruction_embedding.to(device)
        # completed_eval_idx = eval_idx + num_subgoals_to_complete - 1
        completed_eval_idx = eval_idx + num_subgoals_to_complete
        done = 0
        t = 0

        acs = []
        obj_acs = []
        dones = []
        env_rewards = []
        str_act = []
        num_primitive_actions_in_task = sum(
            [
                len(primitive_skill["api_action"])
                for primitive_skill in target_composite_skill_info["primitive_skills"]
            ]
        )
        MAX_STEPS = num_primitive_actions_in_task * EVAL_STEP_RATIO + len(
            expert_init_actions
        )
        # MAX_STEPS = DEFAULT_NUM_STEPS + len(expert_init_actions)
        while not done and t < MAX_STEPS:
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
                    print("expert initialization failed, re-sampling")
                    return eval_policy(
                        env,
                        model,
                        sentence_embedder,
                        visual_preprocessor,
                        max_skill_length,
                        device,
                        scene_type,
                        num_scenes_to_sample,
                        deterministic,
                        log_video,
                        composite_skill_list,  # not used here, but kept for consistency
                        target_composite_skill_info,
                        epsilon=0.1,
                        use_combined_primitive_instructions=USE_COMBINED_PRIMITIVE_INSTRUCTIONS,
                    )
                # _, _ = env.get_transition_reward() # REMOVED FOR NEW EVAL, BUT NEEDED FOR OLD
            else:
                if (len(expert_init_actions) == 0 and t == 0) or (len(expert_init_actions)!= 0 and t == len(expert_init_actions)):
                    video_frames = []
                    value_predict = []
                    env.fixed_eval_set_subskill_type(prev_skill_info)
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
                    value_output = value_output.squeeze().cpu().detach().numpy()
                    value_predict.append(value_output)
                try:
                    _, _ = env.to_thor_api_exec(action, output_object, smooth_nav=True)
                except Exception as e:
                    pass
                next_frame = np.uint8(env.last_event.frame)
                next_frame = Image.fromarray(next_frame)
                next_frame = visual_preprocessor.featurize([next_frame], batch=1).to(
                    device
                )

                feat["frames_buffer"] = torch.cat(
                    [feat["frames_buffer"], next_frame.unsqueeze(0)], dim=1
                ).to(device)
                tensor_action = torch.tensor(
                    vocab["action_low"].word2index(action) - 2
                ).to(device)

                acs.append(tensor_action.cpu())
                obj_index = load_object_class(vocab_obj, output_object)
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
                str_act.append(
                    dict(
                        action=action,
                        object=output_object.split("|")[0]
                        if output_object is not None
                        else None,
                    )
                )
                obj_acs.append(obj_index)

                # MUST call get_transition_reward to update the environment
                # _, _ = env.get_transition_reward()
                # curr_subgoal_idx = env.get_subgoal_idx()
                # print("curr_subgoal_idx",curr_subgoal_idx)
                reward = env.test_get_done()
                if reward == 1:
                    eval_idx += 1
                    # MAX_STEPS = t + 1 + DEFAULT_NUM_STEPS
                    if eval_idx == completed_eval_idx:
                        done = 1
                    else:
                        prev_skill_info = generate_skill_info(traj_data, eval_idx)
                        env.fixed_eval_set_subskill_type(prev_skill_info)
                # elif curr_subgoal_idx == eval_idx:
                #    eval_idx += 1
                env_rewards.append(reward)

            t = t + 1
        subgoal_last_frame_video = np.uint8(env.last_event.frame)
        video_frames.append(subgoal_last_frame_video)
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
            value_output = value_output.squeeze().cpu().detach().numpy()
            value_predict.append(value_output)

        if log_video:
            video_frames = generate_video(
                value_predict, str_act, video_frames, env_rewards
            )

        dones = torch.zeros(len(env_rewards))
        dones[-1] = done
        # vid_caption = f"{composite_instruction[0]}: {'SUCCESS' if done else 'FAIL'}. Completed {curr_subgoal_idx + 1 - original_eval_idx}/{num_subgoals_to_complete} subgoals."
        vid_caption = f"{composite_instruction[0]}: {'SUCCESS' if done else 'FAIL'}. Completed {eval_idx - original_eval_idx}/{num_subgoals_to_complete} subgoals."
        return dict(
            rews=torch.tensor(env_rewards, dtype=torch.float),
            dones=dones,
            video_frames=video_frames if log_video else None,
            video_caption=vid_caption,
            num_primitive_skills_attempted=num_subgoals_to_complete,  # logging misnomer but that's fine
        )


def eval_policy_multi_process(
    ret_queue,
    task_queue,
    offline_rl_model,
    sentence_embedder,
    llm,  # not used, just for compatibility
    resnet,
    config,
    device,
):
    env = ThorEnv()
    while True:
        task_args = task_queue.get()
        if task_args is None:
            break
        ret_queue.put(
            eval_policy(
                env,
                offline_rl_model,
                sentence_embedder,
                resnet,
                config.max_skill_length,
                device,
                *task_args,
            )
        )
    env.stop()

