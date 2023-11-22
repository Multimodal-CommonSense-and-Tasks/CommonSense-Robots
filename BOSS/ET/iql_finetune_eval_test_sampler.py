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
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from ET.et_iql_feeder import remove_spaces_and_lower, numericalize
from iql_finetune_eval_test_sampler import setup_scene, sample_task
from ET.iql_sampler_et import get_action_from_agent
from eval_utils import process_skill_strings
from utils import AttrDict, load_object_class

vocab = torch.load("pp.vocab")
vocab_obj = torch.load("data/obj_cls.vocab")


MAX_FAILS = 1
# MAX_STEPS = 30 * 3
MIN_STEPS_PER_SUBGOAL = 40 # for training
EVAL_STEP_RATIO = 5
#TRAIN_STEP_RATIO = 2
DATA_PATH = "data/json_2.1.0_merge_goto/preprocess"
VISUAL_MODEL = "resnet18"
REWARD_CONFIG = "models/config/rewards.json"

global_task_args = AttrDict()
global_task_args.reward_config = REWARD_CONFIG
global_task_args.visual_model = VISUAL_MODEL
# global_task_args.max_steps = MAX_STEPS


def process_annotation(annotation, model):
    ann_l = revtok.tokenize(remove_spaces_and_lower(annotation))
    ann_l = [w.strip().lower() for w in ann_l]
    ann_token = numericalize(model.vocab_word, ann_l, train=False)
    ann_token = torch.tensor(ann_token).long()
    return ann_token


def run_policy(
    env,
    model,
    visual_preprocessor,
    device,
    subgoal_pool,
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
    epsilon=0.0,
    selected_specific_subgoal=None,
):
    model.eval()
    with torch.no_grad():
        eval_idx, traj_data, r_idx, actually_selected_subgoal = sample_task(
            eval_split,
            subgoal_pool,
            num_subgoals_in_pool,
            selected_specific_subgoal
        )
        num_subgoals_to_complete = len(
            subgoal_pool[actually_selected_subgoal]["primitive_skills"]
        )
        setup_scene(env, traj_data, r_idx, global_task_args)

        expert_init_actions = [
            a["discrete_action"]
            for a in traj_data["plan"]["low_actions"]
            if a["high_idx"] < eval_idx
        ]

        all_instructions = process_skill_strings(
            [
                subgoal_pool[actually_selected_subgoal]["primitive_skills"][i][
                    "annotations"
                ]
                for i in range(
                    len(subgoal_pool[actually_selected_subgoal]["primitive_skills"])
                )
            ]
            + [subgoal_pool[actually_selected_subgoal]["annotation"]]
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

        total_num_subgoals = len(
            subgoal_pool[actually_selected_subgoal]["primitive_skills"]
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
        num_primitive_steps_in_task = sum(
            [
                len(primitive_skill["api_action"])
                for primitive_skill in subgoal_pool[actually_selected_subgoal][
                    "primitive_skills"
                ]
            ]
        )
        num_subgoals_in_task = len(subgoal_pool[actually_selected_subgoal]["primitive_skills"])
        if eval:
            MAX_STEPS = num_primitive_steps_in_task * EVAL_STEP_RATIO
        else:
            MAX_STEPS = MIN_STEPS_PER_SUBGOAL * num_subgoals_in_task #num_primitive_steps_in_task * TRAIN_STEP_RATIO
        #MAX_STEPS = max(MIN_STEPS, MAX_STEPS)
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
                        partial_success * 1 / total_num_subgoals
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
            unnormalized_rews=rewards * total_num_subgoals
            if use_fractional_rewards
            else rewards,
            skill_length=total_num_subgoals,
        )


def run_policy_multi_process(
    ret_queue,
    task_queue,
    offline_rl_model,
    resnet,
    device,
    subgoal_pool,
):
    env = ThorEnv()
    while True:
        task_args = task_queue.get()
        if task_args is None:
            break
        ret_queue.put(
            run_policy(
                env,
                offline_rl_model,
                resnet,
                device,
                subgoal_pool,
                *task_args,
            )
        )
    env.stop()
