import numpy as np
import random
import torch
import os
import sys
from ET.eval_fixed_scene_rollout import eval_policy
from utils import generate_video

from env.online_thor_env import OnlineThorEnv

path = "."
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(path, "gen"))
sys.path.append(os.path.join(path, "models"))
sys.path.append(os.path.join(path, "models", "eval"))
from ET.iql_sampler_et import get_action_from_agent
from utils import AttrDict, load_object_class

vocab = torch.load("pp.vocab")
vocab_obj = torch.load("data/obj_cls.vocab")


def run_policy(
    env,
    model,
    scene_type,
    num_scenes_to_sample,
    deterministic,
    log_video,
    composite_skill_list,
    target_composite_skill=None,
    epsilon=0.1,
    rollout_mode="train",
):
    model.eval()
    with torch.no_grad():
        # format of skills in composite_skill_list is a dictionary with the following keys:
        # "skill": Skill Object
        # "scene_index": int of specific string index
        # "init_action": initial eval action

        model_input = {}
        obss = []
        chained_skill_annotations = []
        acs = []
        obj_acs = []
        env_rewards = []
        str_act = []
        video_frames = []
        value_predict = []
        new_skill_values = []
        new_skill_llm_probs = []
        primitive_skill_types = []
        per_step_primitive_skill_types = []
        skill_switch_points = []
        sampled_skill_types = []
        valid_masks = []
        # sorted_llm_annotations_logprob_dicts = []
        sampled_skill_llm_probs = []
        if rollout_mode == "train_composite":
            target_composite_skill = random.choice(composite_skill_list)

        if target_composite_skill is None:
            # training mode
            obs, lang_annotation, info = env.reset(
                composite_skill_list=composite_skill_list,
                num_scenes_to_sample=num_scenes_to_sample,
            )
        else:
            if rollout_mode == "train_composite":
                obs, lang_annotation, info = env.eval_reset(
                    specific_skill=target_composite_skill, eval=False,
                )
            # testing mode
            else:
                obs, lang_annotation, info = env.eval_reset(
                    specific_skill=target_composite_skill, eval=True
                )
        model_input["frames_buffer"], model_input["language_ann"] = (
            obs,
            lang_annotation,
        )
        model_input["action_traj"] = torch.zeros(1, 0).long().to(env.device)
        model_input["object_traj"] = torch.zeros(1, 0).long().to(env.device)
        curr_frame = info["frame"]
        sampled_scene_index = info["scene_index"]
        if "values" in info:
            new_skill_values.append(info["values"].tolist())
            new_skill_llm_probs.append(info["llm_probs"].tolist())
            sampled_skill_types.append(info["sampled_skill_types"])
        if "primitive_skill_type" in info:
            primitive_skill_types.append(info["primitive_skill_type"])
        if "valid_mask" in info:
            valid_masks.append(info["valid_mask"])
        # if "sorted_llm_annotations_logprob_dict" in info:
        #    sorted_llm_annotations_logprob_dicts.append(
        #        info["sorted_llm_annotations_logprob_dict"]
        #    )
        if "sampled_skill_llm_prob" in info:
            sampled_skill_llm_probs.append(info["sampled_skill_llm_prob"])
        init_action = None
        if "init_action" in info:
            init_action = info["init_action"]
        done = False
        skill_switched = 0
        current_skill_embedding = (
            info["current_skill"].composite_language_embedding.detach().cpu()
        )
        first_skill_length = info["current_skill"].num_skills
        while not done and not info["time_limit_reached"]:
            obss.append(obs[:, -1].cpu().detach())
            chained_skill_annotations.append(current_skill_embedding)
            skill_switch_points.append(skill_switched)
            video_frames.append(curr_frame)
            (action, output_object, value_output) = get_action_from_agent(
                model,
                model_input,
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

            # environment step
            (next_obs, next_lang_embedding), reward, done, info = env.gym_step(
                action, output_object
            )
            next_frame = info["frame"]

            # update the feature buffer
            model_input["frames_buffer"] = next_obs
            model_input["language_ann"] = next_lang_embedding

            # update actions
            tensor_action = torch.tensor(vocab["action_low"].word2index(action) - 2).to(
                env.device
            )
            acs.append(tensor_action.cpu())
            obj_index = load_object_class(vocab_obj, output_object)
            obj_acs.append(obj_index)

            model_input["action_traj"] = torch.cat(
                [model_input["action_traj"], tensor_action.unsqueeze(0).unsqueeze(0)],
                dim=1,
            )
            model_input["object_traj"] = torch.cat(
                [
                    model_input["object_traj"],
                    torch.tensor(obj_index).unsqueeze(0).unsqueeze(0).to(env.device),
                ],
                dim=1,
            )
            model_input["action_traj"] = model_input["action_traj"][
                :, -env.obs_concat_length + 1 :
            ]
            model_input["object_traj"] = model_input["object_traj"][
                :, -env.obs_concat_length + 1 :
            ]
            # we have reset the frame buffer, we need to reset the corresponding action buffers
            if next_obs.shape[1] == 1:
                model_input["action_traj"] = torch.zeros(1, 0).long().to(env.device)
                model_input["object_traj"] = torch.zeros(1, 0).long().to(env.device)
            # next frames
            curr_frame = next_frame
            obs = next_obs

            # save things
            if "primitive_skill_type" in info:
                primitive_skill_types.append(info["primitive_skill_type"])
            str_act.append(
                dict(
                    action=action,
                    object=output_object.split("|")[0]
                    if output_object is not None
                    else None,
                )
            )
            env_rewards.append(reward)
            if "values" in info:
                new_skill_values.append(info["values"].tolist())
                new_skill_llm_probs.append(info["llm_probs"].tolist())
                sampled_skill_types.append(info["sampled_skill_types"])
            if "valid_mask" in info:
                valid_masks.append(info["valid_mask"])
            # if "sorted_llm_annotations_logprob_dict" in info:
            #    sorted_llm_annotations_logprob_dicts.append(
            #        info["sorted_llm_annotations_logprob_dict"]
            #    )
            if "sampled_skill_llm_prob" in info:
                sampled_skill_llm_probs.append(info["sampled_skill_llm_prob"])
            per_step_primitive_skill_types.append(info["per_step_primitive_skill_type"])
            if info["second_sampled_skill"] is not None:
                # if we have started to chain a new composite skill, we should add the second primtiive/composite skill's language_embedding
                current_skill_embedding = (
                    info["second_sampled_skill"]
                    .composite_language_embedding.detach()
                    .cpu()
                )
                skill_switched = 1
            else:
                skill_switched = 0

        video_frames.append(next_frame)
        (action, output_object, value_output) = get_action_from_agent(
            model,
            model_input,
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
                value_predict,
                str_act,
                video_frames,
                env_rewards,
                primitive_skill_types=per_step_primitive_skill_types,
            )

        obss.append(obs[:, -1].cpu().detach())  # next obs
        rewards = torch.tensor(env_rewards, dtype=torch.float)
        dones = torch.zeros(len(rewards))
        dones[-1] = done
        if info["composite_skill"] is not None:
            language_annotation = None  # we don't have labels for this yet
            language_instruction = " ".join(
                info["composite_skill"].primitive_instructions_to_compose
            )  # we don't have labels for this yet
            language_instruction = f"UNLABELED: {language_instruction}"
            # language_latent = info["composite_skill"].composite_language_embedding
            # language_instruction = info["composite_skill"].composite_language_instruction
        else:
            language_annotation = (
                info["current_skill"].composite_language_embedding.cpu().detach()
            )
            language_instruction = info["current_skill"].composite_language_instruction
        # vid_caption = f"{language_instruction}: {'SUCCESS' if done else 'FAIL'}. Scene: {sampled_scene_index}"
        if target_composite_skill is not None:
            skill_attempt_length = target_composite_skill["skill"].num_skills
        elif info["composite_skill"] is not None:
            skill_attempt_length = info["composite_skill"].num_skills
        else:
            skill_attempt_length = info["current_skill"].num_skills
        vid_caption = f"{language_instruction}: {'SUCCESS' if done else 'FAIL'}. Completed {rewards.sum().int().item()}/{skill_attempt_length} subgoals. Scene: {sampled_scene_index}"
        return dict(
            obs=torch.cat(obss),
            acs=torch.tensor(acs),
            obj_acs=torch.tensor(obj_acs),
            rews=rewards,
            dones=dones,
            lang_ann=language_annotation,
            chained_skill_ann=chained_skill_annotations,
            skill_switch_points=torch.tensor(skill_switch_points),
            video_frames=video_frames if log_video else None,
            video_caption=vid_caption,
            composite_skill_object=info["composite_skill"],
            current_skill_object=info["current_skill"],
            init_action=init_action,
            scene_index=sampled_scene_index,
            primitive_skill_types=primitive_skill_types,
            first_skill_length=first_skill_length,
            second_skill_length=skill_attempt_length - first_skill_length,
            num_primitive_skills_attempted=info["current_skill"].num_skills,
            new_skill_values=new_skill_values,
            new_skill_llm_probs=new_skill_llm_probs,
            new_skill_sampled_types=sampled_skill_types,
            sampled_skill_llm_probs=sampled_skill_llm_probs,
            valid_masks=valid_masks,
        )


def run_policy_multi_process(
    ret_queue,
    task_queue,
    offline_rl_model,
    sentence_encoder,
    semantic_search_model,
    llm,
    resnet,
    config,
    device,
    training_scenes_to_consider,
    primitive_skills_to_use,
    task_lengths,
):
    env = OnlineThorEnv(
        offline_rl_model.critics,
        resnet,
        sentence_encoder,
        device,
        use_llm=config.use_llm_for_next_skill,
        llm_model=llm,
        ignore_percentile=config.ignore_percentile,
        filter_invalid_skills=config.filter_invalid_skills,
        scene_type=config.scene_type,
        training_scenes_to_consider=training_scenes_to_consider,
        llm_logprob_weight=config.llm_logprob_weight,
        obs_concat_length=config.max_skill_length,
        rand_init=config.rand_init,
        num_skills_to_sample=config.num_skills_to_sample,
        value_sampling_temp=config.value_sampling_temp,
        llm_sampling_temp=config.llm_sampling_temp,
        max_composite_skill_length=config.max_composite_skill_length,
        skill_match_with_dataset=config.skill_match_with_dataset,
        num_skill_match_generations=config.num_skill_match_generations,
        use_value_func=config.use_value_func,
        use_value_for_next_skill=config.use_value_for_next_skill,
        using_episodic_transformer=True,
        semantic_search_model=semantic_search_model,
        allow_duplicate_next_skills=config.allow_duplicate_next_skills,
        primitive_skills_to_use=primitive_skills_to_use,
        task_lengths=task_lengths,
        eval_per_task_in_json=config.eval_per_task_in_json,
    )
    while True:
        task_args = task_queue.get()
        if task_args is None:
            break
        rollout_mode = task_args.pop(-1)
        if rollout_mode == "fixed_eval":
            # if "eval" in rollout_mode:
            # need to eval
            ret_queue.put(
                eval_policy(
                    env,
                    offline_rl_model,
                    sentence_encoder,
                    resnet,
                    config.max_skill_length,
                    device,
                    *task_args,
                )
            )
        else:
            ret_queue.put(
                run_policy(env, offline_rl_model, *task_args, rollout_mode=rollout_mode)
            )
    env.stop()
