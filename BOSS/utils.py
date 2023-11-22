import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import random
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import wandb
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

plot_color = {
    # "GotoLocation": "red",
    "PickupObject": "brown",
    "PutObject": "blue",
    "ToggleObject": "orange",
    "SliceObject": "black",
    "CleanObject": "purple",
    "HeatObject": "green",
    "CoolObject": "pink",
    "Composite": "red",
}
primitive_skill_types = [
    # "GotoLocation",
    "PickupObject",
    "PutObject",
    "ToggleObject",
    "SliceObject",
    "CleanObject",
    "HeatObject",
    "CoolObject",
]


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_success,
        dataset_failure,
        count_partial_success,
        old_new_sampling,
        first_dataset_ratio=None,
        use_real_length=False,
    ):
        self.dataset_success = dataset_success
        self.dataset_failure = dataset_failure
        self.count_partial_success = count_partial_success
        self.old_new_sampling = old_new_sampling
        self.skill_to_completion_map = {}
        self.first_dataset_ratio = first_dataset_ratio
        self.use_real_length = use_real_length

    def __getitem__(self, i):
        # return depending on length
        #try:
        if self.first_dataset_ratio is not None:
            # set i ourselves as we don't have a weighted sampler above
            if random.random() < self.first_dataset_ratio:
                i = 0
            else:
                i = 1
        try:
            if len(self.dataset_failure) == 0:
                i = random.randint(0, len(self.dataset_success) - 1)
                return self.dataset_success[i]
            elif len(self.dataset_success) == 0:
                i = random.randint(0, len(self.dataset_failure) - 1)
                return self.dataset_failure[i]
            else:
                if i == 0:
                    # sample random index from first dataset
                    i = random.randint(0, len(self.dataset_success) - 1)
                    return self.dataset_success[i]
                elif i == 1:
                    # sample random index from second dataset
                    i = random.randint(0, len(self.dataset_failure) - 1)
                    return self.dataset_failure[i]
        except Exception as e:
            print("Exception in CombinedDataset: {e}")
            #raise Exception
            return self.__getitem__(i)
        #except Exception as e:
            #print("Exception in CombinedDataset")
            #return self.__getitem__(i)
        # else:
        #     raise ValueError("Both datasets are empty")
        # for d in self.datasets:
        #    if i < len(d):
        #        return d[i]
        #    i -= len(d)

    def __len__(self):
        # return sum([len(d) for d in self.datasets])
        if self.use_real_length:
            return min(len(self.dataset_success),len(self.dataset_failure))
        return 2

    def add_traj_to_buffer(
        self,
        frames,
        actions,
        obj_acs,
        rewards,
        terminals,
        language,
        goal_states=None
    ):
        which_dataset = None
        if self.old_new_sampling:
            which_dataset = self.dataset_failure
        else:
            if (terminals[-1] == 1 and rewards[-1] > 0) or (
                self.count_partial_success and rewards.sum() > 0
            ):
                which_dataset = self.dataset_success
            else:
                which_dataset = self.dataset_failure
        if goal_states is None:
            which_dataset.add_traj_to_buffer(
                frames, actions, obj_acs, rewards, terminals, language
            )
        else:
            which_dataset.add_traj_to_buffer(
                frames, actions, obj_acs, rewards, terminals, language, goal_states
            )

    @property
    def rl_buffer(self):
        return self.dataset_success.rl_buffer, self.dataset_failure.rl_buffer

    @rl_buffer.setter
    def rl_buffer(self, value):
        self.dataset_success.rl_buffer = value[0]
        self.dataset_failure.rl_buffer = value[1]

    def update_skill_to_completion_map(self, lang_instr, dones):
        # append if we successfully completed the skill or not so iql_sampler can figure out whether to embed the new skill or primitive skills
        # as policy goals
        if lang_instr not in self.skill_to_completion_map:
            self.skill_to_completion_map[lang_instr] = []
        self.skill_to_completion_map[lang_instr].append(dones[-1].item())


class CombinedDatasetMLP(CombinedDataset):
    def add_traj_to_buffer(
        self, obs, lang_embedding, next_obs, rews, acs, obj_id_acs, dones
    ):
        if dones[-1] == 1 and rews[-1] == 1:
            self.dataset_success.add_traj_to_buffer(
                obs, lang_embedding, next_obs, rews, acs, obj_id_acs, dones
            )
        else:
            self.dataset_failure.add_traj_to_buffer(
                obs, lang_embedding, next_obs, rews, acs, obj_id_acs, dones
            )

    @property
    def rl_buffer(self):
        return AttrDict(
            succ=self.dataset_success.rl_buffer,
            fail=self.dataset_failure.rl_buffer,
            skill_to_completion_map=self.skill_to_completion_map,
        )

    @rl_buffer.setter
    def rl_buffer(self, value):
        self.dataset_success.rl_buffer = value.succ
        self.dataset_failure.rl_buffer = value.fail
        self.skill_to_completion_map = value.skill_to_completion_map


def generate_primitive_skill_list_from_eval_skill_info_list(
    primitive_eval_skill_info_list,
):
    primitive_skills_to_use = []
    for skill_info in primitive_eval_skill_info_list:
        primitive_skills_to_use.extend(
            [primitive_skill for primitive_skill in skill_info["primitive_skills"]]
        )
    for primitive_skill in primitive_skills_to_use:
        primitive_skill["api_actions"] = primitive_skill[
            "api_action"
        ]  # relabeling since online_reward.py expects api_actions

    def tuplify_dict_of_dicts(d):
        to_tuplify = []
        for k in sorted(d):
            if isinstance(d[k], dict):
                to_tuplify.append((k, tuplify_dict_of_dicts(d[k])))
            elif isinstance(d[k], list):
                inner_tuplify = []
                for item in d[k]:
                    if isinstance(item, list):
                        inner_tuplify.append(tuple(item))
                    else:
                        inner_tuplify.append(item)
                to_tuplify.append(tuple(inner_tuplify))
            else:
                to_tuplify.append((k, d[k]))
        return tuple(to_tuplify)

    # now remove duplicate primitive skills which is a list of dicts of inner dicts
    primitive_skill_set = set()
    unique_primitive_skills_to_use = []
    for primitive_skill in primitive_skills_to_use:
        if tuplify_dict_of_dicts(primitive_skill) not in primitive_skill_set:
            primitive_skill_set.add(tuplify_dict_of_dicts(primitive_skill))
            unique_primitive_skills_to_use.append(primitive_skill)
    return unique_primitive_skills_to_use


def send_to_device_if_not_none(data_dict, entry_name, device):
    if entry_name not in data_dict or data_dict[entry_name] is None:
        return None
    else:
        return data_dict[entry_name].to(device)


def log_rollout_metrics(
    rollout_returns,
    successes,
    extra_info,
    rollout_gifs,
    video_captions,
    composite_skill_list: list[dict],
    config,
):
    # aggregate metrics
    rollout_metrics = dict(
        average_return=np.mean(rollout_returns),
        success=np.mean(successes),
    )
    # if "sorted_llm_annotations_logprob_dict" in extra_info:
    #    with open(
    #        config.experiment_name + "_sorted_llm_annotations_logprob_dict.txt", "a"
    #    ) as f:
    #        f.write(str(extra_info["sorted_llm_annotations_logprob_dict"]))
    #    extra_info.pop("sorted_llm_annotations_logprob_dict")
    if "new_skill_values" in extra_info:
        # make wandb histogram for values
        flattened_values = [
            item for sublist in extra_info["new_skill_values"] for item in sublist
        ]
        flattened_valid_masks = [
            item for sublist in extra_info["valid_masks"] for item in sublist
        ]
        flattened_composite_masks = [
            item for sublist in extra_info["is_composite"] for item in sublist
        ]
        # rollout_metrics["new_skill_value_hist"] = wandb.Histogram(flattened_values,)

        # make scatter plot for values
        flattened_skill_names = [
            item
            for sublist in extra_info["new_skill_sampled_types"]
            for item in sublist
        ]
        rollout_metrics[
            "new_skill_values_dist"
        ] = make_color_coded_primitive_scatter_plot(
            flattened_values,
            flattened_skill_names,
            "Value",
            "Skill Value Scatterplot",
            composite_value_masks=flattened_composite_masks,
            flattened_valid_masks=flattened_valid_masks,
            filter_invalid=config.filter_invalid_skills,
        )
        # make scatter plot for llm probs
    flattened_llm_probs = []
    flattened_corresponding_values_for_llm_probs = []
    flattened_corresponding_valid_masks_for_llm_probs = []
    flattened_corresponding_composite_masks_for_llm_probs = []
    flattened_llm_skill_names = []
    if "new_skill_llm_probs" in extra_info:
        # make wandb histogram for llm probs
        for i, sublist in enumerate(extra_info["new_skill_llm_probs"]):
            if sublist is not None:
                flattened_llm_probs.extend(sublist)
                flattened_llm_skill_names.extend(
                    extra_info["new_skill_sampled_types"][i]
                )
                flattened_corresponding_values_for_llm_probs.extend(
                    extra_info["new_skill_values"][i]
                )
                flattened_corresponding_valid_masks_for_llm_probs.extend(
                    extra_info["valid_masks"][i]
                )
                # bug here with index error, i don't know why
                # flattened_corresponding_composite_masks_for_llm_probs.extend(
                #    extra_info["is_composite"][i]
                # )
        if len(flattened_llm_probs) > 0:
            # rollout_metrics["new_skill_llm_prob_hist"] = wandb.Histogram(
            #    flattened_llm_probs,
            # )
            rollout_metrics[
                "new_skill_llm_dist"
            ] = make_color_coded_primitive_scatter_plot(
                flattened_llm_probs,
                flattened_llm_skill_names,
                "LLM Probability",
                "Skill LLM Prob Scatterplot",
                # composite_value_masks=flattened_corresponding_composite_masks_for_llm_probs,
                flattened_valid_masks=flattened_corresponding_valid_masks_for_llm_probs,
                filter_invalid=config.filter_invalid_skills,
            )
        # replace with this in extra_info because of the nones
        extra_info["new_skill_llm_probs"] = flattened_llm_probs
        # plot bar chart of sampled new skill types llm prob distribution
        skill_type_to_llm_probs = defaultdict(list)
        for skill_type, llm_probs in zip(
            flattened_llm_skill_names, flattened_llm_probs
        ):
            skill_type_to_llm_probs[skill_type].append(llm_probs)
        for primitive_skill_type in primitive_skill_types:
            if len(skill_type_to_llm_probs[primitive_skill_type]) > 0:
                skill_type_to_llm_probs[primitive_skill_type] = np.mean(
                    skill_type_to_llm_probs[primitive_skill_type]
                )
            else:
                skill_type_to_llm_probs[primitive_skill_type] = 0
        # llm_bar_chart_data = np.array(
        #    [
        #        skill_type_to_llm_probs[skill_type]
        #        for skill_type in primitive_skill_types
        #    ]
        # )
        # rollout_metrics[
        #    f"sampled_primitive_skill_llm_prob_distribution"
        # ] = make_bar_chart(
        #    values=llm_bar_chart_data,
        #    labels=primitive_skill_types,
        #    title="Primitive Skill LLM Prob Distribution",
        #    xlabel="Skill Type",
        #    ylabel="Average LLM Prob",
        #    ylim=None,
        # )
        # plot bar chart of sampled new skill types llm prob distribution
        skill_probability_dist = torch.softmax(
            (
                torch.tensor(flattened_corresponding_values_for_llm_probs)
                / config.value_sampling_temp
            )
            ** (1 - config.llm_logprob_weight)
            * (torch.tensor(flattened_llm_probs) / config.llm_sampling_temp)
            ** config.llm_logprob_weight,
            dim=0,
        ).numpy()
        if config.filter_invalid_skills:
            skill_probability_dist[
                ~np.array(flattened_corresponding_valid_masks_for_llm_probs).astype(
                    bool
                )
            ] = 0
        skill_type_to_vf_times_llm_probs = defaultdict(list)
        for skill_type, vf_times_llm_probs in zip(
            flattened_llm_skill_names, skill_probability_dist
        ):
            skill_type_to_vf_times_llm_probs[skill_type].append(vf_times_llm_probs)
        for primitive_skill_type in primitive_skill_types:
            if len(skill_type_to_vf_times_llm_probs[primitive_skill_type]) > 0:
                skill_type_to_vf_times_llm_probs[primitive_skill_type] = np.mean(
                    skill_type_to_vf_times_llm_probs[primitive_skill_type]
                )
            else:
                skill_type_to_vf_times_llm_probs[primitive_skill_type] = 0
        # vf_times_llm_bar_chart_data = np.array(
        #    [
        #        skill_type_to_vf_times_llm_probs[skill_type]
        #        for skill_type in primitive_skill_types
        #    ]
        # )
        # rollout_metrics[
        #    f"sampled_primitive_skill_value_times_llm_prob_distribution"
        # ] = make_bar_chart(
        #    values=vf_times_llm_bar_chart_data,
        #    labels=primitive_skill_types,
        #    title="Primitive Skill Value * LLM Prob Distribution",
        #    xlabel="Skill Type",
        #    ylabel="Average Value^(1-weight) * LLM Prob^(weight)",
        #    ylim=None,
        # )
        if len(flattened_llm_skill_names) > 0:
            rollout_metrics[
                "new_skill_values_times_llm_prob_dist"
            ] = make_color_coded_primitive_scatter_plot(
                skill_probability_dist,
                flattened_llm_skill_names,
                f"Skill Value^({1 - config.llm_logprob_weight:0.2f}) * LLM Prob^{config.llm_logprob_weight:0.2f}, Value Temp: {config.value_sampling_temp:0.2f} LLM Temp: {config.llm_sampling_temp:0.2f}",
                "Skill Value * LLM Prob Scatterplot",
                composite_value_masks=flattened_corresponding_composite_masks_for_llm_probs,
                flattened_valid_masks=flattened_corresponding_valid_masks_for_llm_probs,
                filter_invalid=config.filter_invalid_skills,
            )
    num_primitive_skills_attempted = extra_info["num_primitive_skills_attempted"]
    # these are None if no new skills were sampled, otherwise one less than num_primitive_skills_attempted
    # num_primitive_skills_completed = extra_info["num_primitive_skills_completed"]
    # generate per-length return and subgoal success data
    per_number_return = defaultdict(list)
    per_number_success = defaultdict(list)
    # for i, (num_attempts, num_completes) in enumerate(
    #    zip(num_primitive_skills_attempted, num_primitive_skills_completed)
    # ):
    for i, num_attempts in enumerate(num_primitive_skills_attempted):
        per_number_return[num_attempts].append(rollout_returns[i])
        per_number_success[num_attempts].append(successes[i])
    for num_attempts, returns in per_number_return.items():
        # log the averages
        rollout_metrics[f"length_{num_attempts}_return"] = np.mean(returns)
        rollout_metrics[f"length_{num_attempts}_success"] = np.mean(
            per_number_success[num_attempts]
        )
        # log a histogram to wandb
        rollout_metrics[f"length_{num_attempts}_return_distribution"] = make_bar_chart(
            values=np.array(returns),
            labels=list(range(len(returns))),
            title=f"Length {num_attempts} Return Distribution",
            xlabel="Which Task",
            ylabel="Return",
            ylim=None,
        )
        rollout_metrics[f"length_{num_attempts}_success_dist"] = make_bar_chart(
            values=np.array(per_number_success[num_attempts]),
            labels=list(range(len(per_number_success[num_attempts]))),
            title=f"Length {num_attempts} Success Distribution",
            xlabel="Which Task",
            ylabel="Success",
            ylim=None,
        )
    if "first_skill_length" in extra_info:
        first_skill_lengths = extra_info["first_skill_length"]
        second_skill_lengths = extra_info["second_skill_length"]
        first_lengths_to_returns = defaultdict(list)
        first_lengths_to_successes = defaultdict(list)
        second_lengths_to_returns = defaultdict(list)
        second_lengths_to_successes = defaultdict(list)
        for i, (first_skill_length, second_skill_length) in enumerate(
            zip(first_skill_lengths, second_skill_lengths)
        ):
            first_lengths_to_returns[first_skill_length].append(
                extra_info["first_skill_return"][i]
            )
            first_lengths_to_successes[first_skill_length].append(
                extra_info["first_skill_success"][i]
            )
            if second_skill_length is not None:
                second_lengths_to_returns[second_skill_length].append(
                    extra_info["second_skill_return"][i]
                )
                second_lengths_to_successes[second_skill_length].append(
                    extra_info["second_skill_success"][i]
                )
        for first_skill_length, returns in first_lengths_to_returns.items():
            rollout_metrics[
                f"first_skill_length_{first_skill_length}_return"
            ] = np.mean(returns)
            rollout_metrics[
                f"first_skill_length_{first_skill_length}_success"
            ] = np.mean(first_lengths_to_successes[first_skill_length])
        # also do overall success
        rollout_metrics["first_skill_success"] = np.mean(
            extra_info["first_skill_success"]
        )
        for second_skill_length, returns in second_lengths_to_returns.items():
            rollout_metrics[
                f"second_skill_length_{second_skill_length}_return"
            ] = np.mean(returns)
            rollout_metrics[
                f"second_skill_length_{second_skill_length}_success"
            ] = np.mean(second_lengths_to_successes[second_skill_length])
        # also do overall success
        if len(second_lengths_to_returns) > 0:
            rollout_metrics["second_skill_success"] = np.mean(
                [np.mean(v) for v in second_lengths_to_returns.values()]
            )
        extra_info.pop("first_skill_length")
        extra_info.pop("first_skill_return")
        extra_info.pop("first_skill_success")
        extra_info.pop("second_skill_length")
        extra_info.pop("second_skill_return")
        extra_info.pop("second_skill_success")
    # extra_info["num_primitive_skills_completed"] = [
    #    x for x in num_primitive_skills_completed if x is not None
    # ]

    # log composite skill original and new names to wandb table
    composite_skill_name_data = []
    for skill_dict in composite_skill_list:
        skill = skill_dict["skill"]
        composed_primitives = " ".join(skill.primitive_instructions_to_compose)
        joined_name = skill.composite_language_instruction
        llm_prob = skill.llm_prob
        scene_index = skill_dict["scene_index"]
        length = skill.num_skills
        composite_skill_name_data.append(
            [composed_primitives, joined_name, llm_prob, scene_index, length]
        )
    table = wandb.Table(
        columns=["Primitives", "Name", "LLM Prob", "Scene Index", "Length"],
        data=composite_skill_name_data,
    )
    # wandb.log({"composite_skill_table": table})
    rollout_metrics["composite_skill_table"] = table

    for key, value in extra_info.items():
        if key == "primitive_skill_types":
            counted_skills = Counter(value)
            data = np.array(
                [counted_skills[skill_type] for skill_type in primitive_skill_types]
            )
            data = data / data.sum()
            # plt bar chart because wandb bar chart doesn't update
            rollout_metrics[f"used_primitive_skill_dist"] = make_bar_chart(
                values=data,
                labels=primitive_skill_types,
                title="Used Primitive Skill Distribution",
                xlabel="Skill Type",
                ylabel="Frequency",
                ylim=(0, 1),
            )
        elif key == "new_skill_sampled_types":
            # plot bar chart of sampled new skill types frequency distribution
            # then plot bar chart of their average values, llm probs, and the product of the two
            vf_values = extra_info["new_skill_values"]
            frequency_counter = Counter()

            skill_type_to_average_value = defaultdict(list)
            # compute average value and llm prob for each skill type
            for i, sample in enumerate(value):
                for j, skill_type in enumerate(sample):
                    skill_type_to_average_value[skill_type].append(vf_values[i][j])
                sample_counter = Counter(sample)
                frequency_counter.update(sample_counter)
            for key in primitive_skill_types:
                if len(skill_type_to_average_value[key]) > 0:
                    skill_type_to_average_value[key] = np.mean(
                        skill_type_to_average_value[key]
                    )
                else:
                    skill_type_to_average_value[key] = 0

            # plot bar chart of sampled new skill types frequency distribution
            freq_data = np.array(
                [frequency_counter[skill_type] for skill_type in primitive_skill_types]
            )
            freq_data = freq_data / freq_data.sum()
            rollout_metrics[f"sampled_primitive_skill_distribution"] = make_bar_chart(
                values=freq_data,
                labels=primitive_skill_types,
                title="Sampled Primitive Skill Distribution",
                xlabel="Skill Type",
                ylabel="Frequency",
                ylim=(0, 1),
            )

            # plot bar chart of sampled new skill types average value distribution
            # vf_data = np.array(
            #    [
            #        skill_type_to_average_value[skill_type]
            #        for skill_type in primitive_skill_types
            #    ]
            # )
            # rollout_metrics[
            #    f"sampled_primitive_skill_value_distribution"
            # ] = make_bar_chart(
            #    values=vf_data,
            #    labels=primitive_skill_types,
            #    title="Primitive Skill Average Value Distribution",
            #    xlabel="Skill Type",
            #    ylabel="Average Value",
            #    ylim=None,
            # )

        else:
            if len(value) > 0 and not isinstance(value[0], str):
                rollout_metrics[f"{key} Mean"] = np.mean(value)
                rollout_metrics[f"{key} Min"] = np.mean(np.min(value, axis=-1))
                rollout_metrics[f"{key} Max"] = np.mean(np.max(value, axis=-1))
            else:
                print(f"warning: {key} has no values")
            # normalized_values = value / np.sum(value, axis=-1, keepdims=True)
            # normalize entropy by the length of the inner dimension
            # rollout_metrics[f"{key} Entropy"] = np.mean(
            #    (-normalized_values * np.log(normalized_values)).sum(axis=-1)
            # )
            # if len(normalized_values.shape) > 1:
            #    rollout_metrics[f"{key} Length-Normalized Entropy"] = (
            #        rollout_metrics[f"{key} Entropy"] / normalized_values.shape[1]
            #    )

    # if "eval" in rollout_mode:
    if len(rollout_gifs) > 0:
        # sort both rollout_gifs and video_captions by the caption so that we have a consistent ordering
        rollout_gifs, video_captions = zip(
            *sorted(zip(rollout_gifs, video_captions), key=lambda x: x[1])
        )
        for i, (gif, caption) in enumerate(zip(rollout_gifs, video_captions)):
            rollout_metrics["videos_%d" % i] = wandb.Video(
                gif, caption=caption, fps=3, format="mp4"
            )
    return rollout_metrics


def make_bar_chart(values, labels, title, xlabel, ylabel, ylim):
    """
    make a bar chart from a list of values and labels
    """

    # plt.figure(figsize=(10, 5))
    # plt.bar(labels, values)
    plt.bar(range(len(values)), values, tick_label=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    ax = plt.gca()
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45)
    plt.tight_layout()
    ret_wandb_image = wandb.Image(plt)
    plt.close()
    plt.cla()
    plt.clf()
    return ret_wandb_image


def make_color_coded_primitive_scatter_plot(
    flattened_values,
    flattened_skill_names,
    x_label,
    title,
    composite_value_masks=[],
    flattened_valid_masks=[],
    filter_invalid=False,
):
    value_counting = dict()
    for i in range(len(flattened_values)):
        value = flattened_values[i]
        skill_name = flattened_skill_names[i]
        valid = (
            1 if len(flattened_valid_masks) == 0 else float(flattened_valid_masks[i])
        )
        is_composite = (
            0 if len(composite_value_masks) == 0 else float(composite_value_masks[i])
        )
        # if skill_name.lower() == "gotolocation":
        #    skill = random.uniform(0.0, 1.0)
        #    point = np.array([value, skill]).reshape(1, -1)
        if skill_name.lower() == "pickupobject":
            skill = random.uniform(0, 1.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "putobject":
            skill = random.uniform(1.0, 2.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "toggleobject":
            skill = random.uniform(2.0, 3.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "sliceobject":
            skill = random.uniform(3.0, 4.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "heatobject":
            skill = random.uniform(4.0, 5.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "coolobject":
            skill = random.uniform(5.0, 6.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "cleanobject":
            skill = random.uniform(6.0, 7.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "composite":
            skill = random.uniform(7.0, 8.0)
            point = np.array([value, skill, valid, True]).reshape(1, -1)
        if skill_name not in value_counting:
            value_counting[skill_name] = point
        else:
            value_counting[skill_name] = np.concatenate(
                (value_counting[skill_name], point), axis=0
            )
    return make_scatter_plot(value_counting, plot_color, x_label, title, filter_invalid)


def make_scatter_plot(
    key_to_value: dict,
    color_map: dict,
    x_label: str,
    title: str,
    filter_invalid: bool,
):
    fig, ax = plt.subplots()
    # first, aggregate all values to get the percentiles
    all_values = []
    for key in key_to_value:
        if filter_invalid:
            valid_mask = key_to_value[key][:, 2].astype(bool)
            all_values.append(key_to_value[key][valid_mask, 0])
        else:
            all_values.append(key_to_value[key][:, 0])
    all_values = np.concatenate(all_values, axis=0)
    percentiles = np.percentile(all_values, [10, 25, 50, 75, 90])
    # then, plot the percentiles with vertical dotted lines
    for i in range(len(percentiles)):
        ax.axvline(percentiles[i], color="black", linestyle="dotted", alpha=0.5)
    # then, plot the actual data
    for skill_name in key_to_value:
        color = color_map[skill_name]
        data = key_to_value[skill_name]
        value = data[:, 0]
        skill = data[:, 1]
        valid = data[:, 2].astype(bool)
        is_composite = data[:, 3].astype(bool)
        # scatter plot all valid primitive skills
        ax.scatter(
            value[valid & ~is_composite],
            skill[valid & ~is_composite],
            label=skill_name,
            c=color,
            s=10,
            alpha=0.3,
        )
        # scatter plot all invalid skills with marker x
        ax.scatter(
            value[~valid & ~is_composite],
            skill[~valid & ~is_composite],
            c=color,
            s=10,
            alpha=0.3,
            marker="x",
        )
        # scatter plot all valid composite skills with more alpha
        ax.scatter(
            value[valid & is_composite],
            skill[valid & is_composite],
            c=color,
            s=10,
            alpha=1.0,
            marker="*",
        )
        # scatter plot copmosite and invalid skills with marker x and more alpha
        ax.scatter(
            value[~valid & is_composite],
            skill[~valid & is_composite],
            c=color,
            s=10,
            alpha=1.0,
            marker="X",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Skill")
    ax.legend(bbox_to_anchor=(0.05, 1.0))
    ax.set_title(title)
    plt.tight_layout()
    ret_wandb_image = wandb.Image(fig)
    plt.close()
    plt.cla()
    plt.clf()
    return ret_wandb_image


def generate_video(
    value_predictions, str_act, video_frames, env_rewards, primitive_skill_types=None
):
    value_font = ImageFont.truetype("FreeMono.ttf", 20)
    action_font = ImageFont.truetype("FreeMono.ttf", 14)
    gif_logs = []
    for frame_number in range(len(video_frames)):
        img = video_frames[frame_number]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        if frame_number != 0:
            if len(env_rewards) > 0:
                reward_log = env_rewards[frame_number - 1]
                draw.text(
                    (1, 260),
                    "Reward: %.1f" % (reward_log),
                    fill=(255, 255, 255),
                    font=value_font,
                )
                return_log = sum(env_rewards[0:frame_number])
                draw.text(
                    (150, 260),
                    "Return: %.1f" % (return_log),
                    fill=(255, 255, 255),
                    font=value_font,
                )

        if frame_number != len(video_frames) - 1:
            if len(str_act) > 0:
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
                if primitive_skill_types is not None:
                    draw.text(
                        (1, 31),
                        f"Skill: {primitive_skill_types[frame_number]}",
                        fill=(255, 255, 255),
                        font=action_font,
                    )
        if len(value_predictions) != 0:
            value_log = value_predictions[frame_number]
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
    return video_frames


def load_object_class(vocab_obj, object_name):
    """
    load object classes for interactive actions
    """
    if object_name is None:
        return 0
    object_class = object_name.split("|")[0]
    return vocab_obj.word2index(object_class)


def extract_item(possible_tensor):
    if isinstance(possible_tensor, torch.Tensor):
        return possible_tensor.item()
    return possible_tensor


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def flip_tensor(tensor, on_zero=1, on_non_zero=0):
    """
    flip 0 and 1 values in tensor
    """
    res = tensor.clone()
    res[tensor == 0] = on_zero
    res[tensor != 0] = on_non_zero
    return res


# def weighted_mask_loss(pred_masks, gt_masks):
#    """
#    mask loss that accounts for weight-imbalance between 0 and 1 pixels
#    """
#    bce = nn.BCEWithLogitsLoss()(pred_masks, gt_masks)
#    flipped_mask = flip_tensor(gt_masks)
#    inside = (bce * gt_masks).sum() / (gt_masks).sum()
#    outside = (bce * flipped_mask).sum() / (flipped_mask).sum()
#    return inside + outside


def obj_classes_loss(pred_obj_cls, gt_obj_cls):
    """
    Compute a cross-entropy loss for the object class predictions.
    """
    # the interaction objects should be non zeros
    assert not (gt_obj_cls == 0).any()
    # compute the loss for interaction objects
    obj_cls_loss = F.cross_entropy(pred_obj_cls, gt_obj_cls, reduction="mean")
    return obj_cls_loss


def weighted_mask_loss(pred_masks, gt_masks):
    """
    mask loss that accounts for weight-imbalance between 0 and 1 pixels
    """
    # pred_mask [batch * len, 1, 300, 300], gt_mask[batch * len, 1, 300, 300]
    bce = nn.BCEWithLogitsLoss(reduction="none")(pred_masks, gt_masks)
    flipped_mask = flip_tensor(gt_masks)
    bs = bce.shape[0]
    bce = bce.view(bs, -1)
    gt_masks = gt_masks.view(bs, -1)
    flipped_mask = flipped_mask.view(bs, -1)
    # per-pixel summation
    inside = torch.sum(bce * gt_masks, dim=1) / (torch.sum(gt_masks, dim=1) + 1e-5)
    outside = torch.sum(bce * flipped_mask, dim=1) / (
        torch.sum(flipped_mask, dim=1) + 1e-5
    )
    return torch.mean(inside + outside)


def weighted_mask_pred_accuracy(pred_masks, gt_masks):
    """
    mask pred accuracy that takes average of 0 and 1 pixels
    """
    bs = pred_masks.shape[0]
    reshaped_gt_masks = gt_masks.reshape(bs, -1)
    flipped_mask = flip_tensor(reshaped_gt_masks)
    rounded_predictions = torch.round(torch.sigmoid(pred_masks)).reshape(bs, -1)
    inside = torch.sum(rounded_predictions * reshaped_gt_masks, dim=1) / (
        torch.sum(reshaped_gt_masks, dim=1) + 1e-5
    )
    outside = torch.sum((1 - rounded_predictions) * flipped_mask, dim=1) / (
        torch.sum(flipped_mask, dim=1) + 1e-5
    )
    # inside = (rounded_predictions * gt_masks).sum() / (gt_masks).sum()
    # outside = ((1 - rounded_predictions) * flipped_mask).sum() / (flipped_mask).sum()
    return 100 * torch.mean(inside + outside) / 2


def concat_videos_across_widths(list_of_np_videos):
    """
    concatenate videos across widths
    """
    # get longest video
    longest_video = max(list_of_np_videos, key=len)
    # get number of frames in longest video
    num_frames = len(longest_video)
    # extend all videos to have the same number of frames
    list_of_np_videos = [
        np.concatenate((v, np.zeros((num_frames - len(v), *v.shape[1:]))))
        for v in list_of_np_videos
    ]
    # concatenate videos across widths, the 3rd dimension is the width
    # wandb expects (time x channels x height x width)
    concatted_vids = np.concatenate(list_of_np_videos, axis=3)
    return concatted_vids


def create_video_grid(list_of_np_videos):
    # get longest video
    longest_video = max(list_of_np_videos, key=len)
    # get number of frames in longest video
    num_frames = len(longest_video)
    # extend all videos to have the same number of frames
    list_of_np_videos = [
        np.concatenate((v, np.zeros((num_frames - len(v), *v.shape[1:]))))
        for v in list_of_np_videos
    ]

    # turn into (time x num_images x channels x height x width)
    batched_videos = np.stack(list_of_np_videos, axis=1)

    num_vids = len(list_of_np_videos)
    num_cols = int(np.ceil(num_vids / 5))
    num_rows = int(np.ceil(num_vids / num_cols))
    # turn to torch grid
    concatted_tensor = torch.from_numpy(batched_videos)

    # turn to grid of tensors
    final_grid_images = []
    for i in range(num_frames):
        grid = torchvision.utils.make_grid(concatted_tensor[i], nrow=num_rows)
        final_grid_images.append(grid.numpy())
    # final_grid_images will be list of images across time of size (channels x new_height x new_width)
    # wandb expects (time x channels x height x width)
    # so stack it to get (time x channels x new_height x new_width)
    return np.stack(final_grid_images, axis=0)


def save_image(pred_mask_masked, batch_segmentation_mask_masked, choose_image=5):

    batch_size = pred_mask_masked.shape[0]
    if batch_size > choose_image:
        choose_index = np.array(random.sample(range(0, batch_size - 1), choose_image))
        pred_mask_masked = torch.round(torch.sigmoid(pred_mask_masked[choose_index]))
        batch_segmentation_mask_masked = batch_segmentation_mask_masked[choose_index]
    catted_images = [batch_segmentation_mask_masked] + [pred_mask_masked]
    image_grid = torchvision.utils.make_grid(catted_images, nrow=choose_image)
    transform = torchvision.transforms.ToPILImage()
    image_grid = transform(torch.cat([im for im in image_grid], dim=0).unsqueeze(0))
    return image_grid


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __deepcopy__(self, memo):
        return AttrDict(copy.deepcopy(dict(self), memo))
