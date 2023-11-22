import numpy as np
from env.online_thor_env import OnlineThorEnv
from ET.eval_fixed_scene_rollout import (
    sample_task,
    setup_scene,
    global_task_args,
    generate_skill_info,
)
from eval_utils import process_skill_strings
import torch
from saycan_llm import SaycanPlanner
import os
import sys
from utils import generate_video

path = "."
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(path, "gen"))
sys.path.append(os.path.join(path, "models"))
sys.path.append(os.path.join(path, "models", "eval"))
from ET.iql_sampler_et import get_action_from_agent
from utils import AttrDict, load_object_class

from PIL import Image

vocab = torch.load("pp.vocab")
vocab_obj = torch.load("data/obj_cls.vocab")

EVAL_STEP_RATIO = 5


def get_next_skill_from_saycan(
    model,
    saycan_planner: SaycanPlanner,
    sentence_embedder,
    high_level_skill: str,
    all_primitive_skills: list[dict],
    already_completed_skills: list[str],
    feat: dict,
    device,
):
    primitive_skill_annotations = process_skill_strings(
        [skill["annotations"] for skill in all_primitive_skills]
    )
    llm_logprobs = saycan_planner.get_saycan_logprobs(
        already_completed_skills,
        primitive_skill_annotations,
        [high_level_skill],
    )
    # get value logprobs
    primitive_embeddings = sentence_embedder.encode(
        primitive_skill_annotations, convert_to_tensor=True
    )
    values = []
    for primitive_embedding in primitive_embeddings:
        primitive_embedding = primitive_embedding.to(device)
        feat["language_ann"] = primitive_embedding.unsqueeze(0)
        *_, value = model.step(feat, ret_value=True)
        values.append(value.unsqueeze(0))
    values = torch.cat(values, dim=0)
    values = torch.clamp(values, min=0, max=1).cpu()
    # combine LLM and values
    llm_probs = torch.exp(llm_logprobs)
    combined_affordance_probs = llm_probs * values
    # now take the argmax
    next_skill_idx = torch.argmax(combined_affordance_probs).item()
    feat["language_ann"] = sentence_embedder.encode(
        primitive_skill_annotations[next_skill_idx : next_skill_idx + 1],
        convert_to_tensor=True,
    ).to(
        device
    )  # re-encode the selected skill so there's no padding
    return primitive_skill_annotations[next_skill_idx]


def eval_policy(
    env,
    model,
    saycan_planner: SaycanPlanner,
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
    epsilon=0,
):
    model.eval()
    with torch.no_grad():
        eval_idx, traj_data, r_idx = sample_task(
            scene_type,
            target_composite_skill_info,
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

        composite_instruction = process_skill_strings(
            [target_composite_skill_info["annotation"]]
        )
        all_primitive_skills = [
            skill for skill in target_composite_skill_info["primitive_skills"]
        ]
        feat = {}
        completed_eval_idx = eval_idx + num_subgoals_to_complete
        done = 0
        t = 0

        acs = []
        obj_acs = []
        dones = []
        env_rewards = []
        str_act = []
        completed_skills = []
        predicted_skills = []
        num_primitive_skills_in_task = sum(
            [
                len(primitive_skill["api_action"])
                for primitive_skill in target_composite_skill_info["primitive_skills"]
            ]
        )
        MAX_STEPS = num_primitive_skills_in_task * EVAL_STEP_RATIO + len(
            expert_init_actions
        )
        while not done and t < MAX_STEPS:
            if (len(expert_init_actions) == 0 and t == 0) or (len(expert_init_actions)!= 0 and t == len(expert_init_actions)):
                curr_image = Image.fromarray(np.uint8(env.last_event.frame))

                curr_frame = visual_preprocessor.featurize([curr_image], batch=1).to(
                    device
                )
                feat["frames_buffer"] = curr_frame.unsqueeze(0).to(device)
                feat["action_traj"] = torch.zeros(1, 0).long().to(device)
                feat["object_traj"] = torch.zeros(1, 0).long().to(device)
                # get first saycan planner action
                selected_skill_annotation = get_next_skill_from_saycan(
                    model,
                    saycan_planner,
                    sentence_embedder,
                    composite_instruction[0],
                    all_primitive_skills,
                    completed_skills,
                    feat,
                    device,
                )
                predicted_skills.append(selected_skill_annotation)

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
                        epsilon=epsilon,
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
                    ret_value=False,
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

                reward = env.test_get_done()
                if reward == 1:
                    # reset buffers as the policy acts only on the primitive skills
                    completed_skills.append(selected_skill_annotation)
                    feat["frames_buffer"] = next_frame.unsqueeze(0).to(device)
                    feat["action_traj"] = torch.zeros(1, 0).long().to(device)
                    feat["object_traj"] = torch.zeros(1, 0).long().to(device)
                    selected_skill_annotation = get_next_skill_from_saycan(
                        model,
                        saycan_planner,
                        sentence_embedder,
                        composite_instruction[0],
                        all_primitive_skills,
                        completed_skills,
                        feat,
                        device,
                    )
                    predicted_skills.append(selected_skill_annotation)
                    eval_idx += 1
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
            ret_value=False,
        )

        if value_output != None:
            value_output = value_output.squeeze().cpu().detach().numpy()
            value_predict.append(value_output)

        if log_video:
            video_frames = generate_video(value_predict, [], video_frames, [])

        dones = torch.zeros(len(env_rewards))
        dones[-1] = done
        # vid_caption = f"{composite_instruction[0]}: {'SUCCESS' if done else 'FAIL'}. Completed {curr_subgoal_idx + 1 - original_eval_idx}/{num_subgoals_to_complete} subgoals."
        vid_caption = f"{composite_instruction[0]}: {'SUCCESS' if done else 'FAIL'}. Completed {eval_idx - original_eval_idx}/{num_subgoals_to_complete} subgoals."
        ground_truth_annotation_sequence = " ".join(
            process_skill_strings(
                [skill["annotations"] for skill in all_primitive_skills]
            )
        )
        return dict(
            rews=torch.tensor(env_rewards, dtype=torch.float),
            dones=dones,
            video_frames=video_frames if log_video else None,
            video_caption=vid_caption,
            num_primitive_skills_attempted=num_subgoals_to_complete,  # logging misnomer but that's fine
            completed_skills=" ".join(completed_skills),
            predicted_skills=" ".join(predicted_skills),
            high_level_skill=composite_instruction[0],
            ground_truth_sequence=ground_truth_annotation_sequence,
        )


def run_policy_multi_process(
    ret_queue,
    task_queue,
    agent_model,
    saycan_planner,
    sentence_encoder,
    resnet,
    config,
    device,
):
    env = OnlineThorEnv(
        agent_model.critics,
        resnet,
        sentence_encoder,
        device,
        obs_concat_length=20,  # arbitrary
        rand_init=False,
        use_llm=False,
        llm_model=None,
        ignore_percentile=None,
        filter_invalid_skills=None,
        scene_type=config.scene_type,
        training_scenes_to_consider=None,
    )
    while True:
        task_args = task_queue.get()
        if task_args is None:
            break
        ret_queue.put(
            eval_policy(
                env,
                agent_model,
                saycan_planner,
                sentence_encoder,
                resnet,
                config.max_skill_length,
                device,
                *task_args,
            )
        )
