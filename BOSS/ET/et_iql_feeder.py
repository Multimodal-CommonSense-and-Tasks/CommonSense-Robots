import os
import pickle
from custom_lmdb_reader import CustomLMDBReader
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import revtok
import pyxis as px
from torch.nn.utils.rnn import pad_sequence
import random
from ET.et_feeder import (
    load_object_class,
    remove_spaces,
    remove_spaces_and_lower,
    numericalize,
    load_pyxis,
    ETCompositeFeeder,
)


class ETIQLFeeder(ETCompositeFeeder):
    def __init__(
        self,
        path,
        data_type,
        sample_primitive_skill,
        max_skill_length,
        use_full_skill=False,
    ):

        self.path = path
        if hasattr(self, "drop_old_data") and self.drop_old_data:  # for RL buffer
            self.data = None
        else:
            self.data = self.load_pyxis()
        self.sample_primitive_skill = sample_primitive_skill
        self.use_full_skill = use_full_skill
        self.vocab_obj = torch.load("ET/obj_cls.vocab")
        self.vocab_ann = torch.load("ET/human.vocab")
        self.max_skill_length = max_skill_length
        self.include_list_dict = [
            "lang_low_action",
            "lang_object_ids",
            "lang_valid_interact",
            "lang_subgoals",
            "lang_combinations",
            "lang_ridx",
        ]

        self.include_all_dict = [
            "traj_resnet_feature",
            "skill_switch_point",
        ]

        if not self.sample_primitive_skill:
            pkl_name = "ET/ET_IQL_composite_skill_set_" + data_type
        else:
            pkl_name = "ET/ET_IQL_primitive_skill_set_" + data_type
        pkl_name += ".pkl"
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                (self.single_sample_trajectory_dict,) = pickle.load(f)
        else:
            self.create_single_sample_trajectory_dict()
            # in case another process takes over and saves it first
            if not os.path.exists(pkl_name):
                with open(pkl_name, "wb") as f:
                    pickle.dump(
                        (self.single_sample_trajectory_dict,), f,
                    )

    def create_single_sample_trajectory_dict(self):
        print("generating pickle file")
        single_sample_trajectory_dict = {}
        total_samples = 0

        if not self.sample_primitive_skill:
            for i in tqdm(range(len(self.data))):

                num_skill = self.data[i]["lang_ridx"].shape[0]

                for j in range(num_skill):
                    single_sample_trajectory_dict[total_samples] = (i, j)
                    total_samples += 1

        else:
            for i in tqdm(range(len(self.data))):
                num_skill = self.data[i]["lang_ridx"].shape[0]
                for j in range(num_skill):
                    if len(self.data[i]["lang_subgoals"][j].split("+")) == 1:
                        single_sample_trajectory_dict[total_samples] = (i, j)
                        total_samples += 1
        self.single_sample_trajectory_dict = single_sample_trajectory_dict

    def get_data_from_pyxis(self, i, j):
        data_dict = self.data[i]
        if "lang_object_ids" in data_dict:
            data_dict["lang_object_ids"] = pickle.loads(data_dict["lang_object_ids"])

        traj_dict = {}
        for key, value in data_dict.items():
            if key in self.include_list_dict:
                traj_dict[key] = value[j]
            elif key in self.include_all_dict:
                traj_dict[key] = value

        skill_dict = {}
        subgoal_idx = [int(x) for x in traj_dict["lang_subgoals"].split("+")]

        # process resnet feature
        skill_start_index = subgoal_idx[0]
        skill_end_index = subgoal_idx[-1]

        start_index = data_dict["skill_switch_point"][skill_start_index]

        if skill_end_index == len(data_dict["skill_switch_point"]) - 1:
            skill_feature = data_dict["traj_resnet_feature"][start_index:]

        else:
            end_index = (
                data_dict["skill_switch_point"][skill_end_index + 1] + 1
            )  # goal state include the last state of skill combination
            skill_feature = data_dict["traj_resnet_feature"][start_index:end_index]
        skill_feature = torch.from_numpy(
            skill_feature
        ).float()  # this includes the last state
        low_action = np.asarray(
            [float(action) for action in traj_dict["lang_low_action"].split("+")]
        )
        low_action = torch.from_numpy(low_action).float()
        # low_action = torch.tensor(float(traj_dict["lang_low_action"].split("+")))
        object_ids = traj_dict["lang_object_ids"]
        object_ids = np.asarray(
            [load_object_class(self.vocab_obj, ids) for ids in object_ids]
        )
        object_ids = torch.from_numpy(object_ids).float()
        valid_interact = np.asarray(
            [int(va) for va in traj_dict["lang_valid_interact"].split("+")]
        )
        valid_interact = torch.from_numpy(valid_interact)

        annotation = traj_dict["lang_combinations"]
        ann_l = revtok.tokenize(remove_spaces_and_lower(annotation))
        ann_l = [w.strip().lower() for w in ann_l]
        ann_token = numericalize(self.vocab_ann, ann_l, train=True)
        ann_token = np.asarray(ann_token)
        ann_token = torch.from_numpy(ann_token).float()

        if self.use_full_skill:
            start = 0
        else:
            start = random.randint(0, len(low_action) - 1)
        skill_dict["skill_feature"] = skill_feature[
            start : start + self.max_skill_length + 1
        ]  # shape:  batch x seq_len x 512 x 7 x 7
        skill_dict["low_action"] = (
            low_action[start : start + self.max_skill_length] - 1
        )  # shape: batch x action_len
        rewards = torch.zeros(skill_dict["low_action"].shape[0])
        if start + self.max_skill_length >= len(skill_dict["low_action"]):
            rewards[-1] = 1
        assert len(skill_dict["skill_feature"]) == len(skill_dict["low_action"]) + 1
        skill_dict["object_ids"] = object_ids[
            start : start + self.max_skill_length
        ]  # shape: batch x object
        skill_dict["valid_interact"] = valid_interact[
            start : start + self.max_skill_length
        ]
        skill_dict["annotation"] = annotation
        skill_dict["ann_token"] = ann_token
        skill_dict["token_length"] = ann_token.shape[0]  # token length
        # skill_dict["skill_length"] = skill_dict["low_action"].shape[0]
        skill_dict["terminal"] = skill_dict["reward"] = rewards
        skill_dict["feature_length"] = skill_dict["skill_feature"].shape[
            0
        ]  # feature length one more than low action number
        return skill_dict


def collate_func(batch_dic):
    batch_len = len(batch_dic)  # size
    # max_feature_length = max(
    #    [dic["feature_length"] for dic in batch_dic]
    # )  # max feature length
    # max_skill_length = max(
    #    [dic["ann_token"] for dic in batch_dic]
    # )  # max skill length / action number

    skill_feature = []
    annotations = []
    low_action = []
    object_ids = []
    valid_interact = []
    ann_token = []
    # skill_length = []
    feature_length = []
    token_length = []
    reward = []
    terminal = []
    for i in range(batch_len):
        dic = batch_dic[i]
        skill_feature.append(dic["skill_feature"])
        low_action.append(dic["low_action"])
        object_ids.append(dic["object_ids"])
        valid_interact.append(dic["valid_interact"])
        ann_token.append(dic["ann_token"])
        # skill_length.append(dic["skill_length"])
        feature_length.append(dic["feature_length"])
        token_length.append(dic["token_length"])
        annotations.append(dic["annotation"])
        reward.append(dic["reward"])
        terminal.append(dic["terminal"])

    res = {}
    res["skill_feature"] = pad_sequence(
        skill_feature, batch_first=True, padding_value=0
    )
    # pad one more to do parallel curr state/next state value computation and match length with skill feature
    low_actions_padded = pad_sequence(low_action, batch_first=True, padding_value=0)
    res["low_action"] = torch.cat(
        (low_actions_padded, torch.zeros((batch_len,)).unsqueeze(1)), dim=1
    )
    obj_ids_padded = pad_sequence(object_ids, batch_first=True, padding_value=0)
    res["object_ids"] = torch.cat(
        (obj_ids_padded, torch.zeros((batch_len,)).unsqueeze(1)), dim=1
    )

    res["valid_interact"] = pad_sequence(
        valid_interact, batch_first=True, padding_value=0
    )
    res["ann_token"] = pad_sequence(ann_token, batch_first=True, padding_value=0)
    # res["skill_length"] = torch.tensor(np.asarray(skill_length))
    res["feature_length"] = torch.tensor(np.asarray(feature_length))
    res["token_length"] = torch.tensor(np.asarray(token_length))
    res["annotation"] = annotations
    res["reward"] = pad_sequence(reward, batch_first=True, padding_value=-1)
    res["terminal"] = pad_sequence(terminal, batch_first=True, padding_value=-1)

    return res


class ETIQLRLFeeder(ETIQLFeeder):
    def __init__(
        self,
        path,
        split,
        use_full_skill,
        max_skill_length,
        drop_old_data,
        simulate_per,
        per_prob=0.2,
    ):
        self.drop_old_data = drop_old_data
        super(ETIQLRLFeeder, self).__init__(
            path,
            split,
            sample_primitive_skill=True,
            max_skill_length=max_skill_length,
            use_full_skill=use_full_skill,
        )
        self.rl_buffer = []
        self.skill_to_completion_map = {}  # for the finetuning script
        self.simulate_per = simulate_per
        self.per_prob = per_prob

    def __len__(self):
        if self.drop_old_data:
            return len(self.rl_buffer)
        else:
            return len(self.single_sample_trajectory_dict) + len(self.rl_buffer)

    def __getitem__(self, index):
        if self.drop_old_data:
            return self.get_data_from_RL_buffer(index)
        else:
            if index < len(self.single_sample_trajectory_dict):
                return super(ETIQLRLFeeder, self).__getitem__(index)
            else:
                return self.get_data_from_RL_buffer(
                    index - len(self.single_sample_trajectory_dict)
                )

    def add_traj_to_buffer(
        self, frames: torch.Tensor, actions: torch.Tensor, obj_acs: torch.Tensor, rewards: torch.Tensor, terminals: torch.Tensor, language: torch.Tensor,
    ):
        assert len(actions) == len(frames) - 1
        assert len(actions) == len(obj_acs)
        add_to_buffer = True and not torch.all(language == 0)
        if len(language.shape) > 1:
            language = language.squeeze(0)
        if self.simulate_per:
            if (
                rewards[-1] != 1 and random.random() > self.per_prob
            ):  # if failure but we are adding to buffer anyway
                add_to_buffer = False
        if add_to_buffer:
            # all sentences end in period, get its index of 38 and don't use anything after it
            if language.shape[0] != 768: # not a vector
                language = language[: len(language) - language.tolist()[::-1].index(38)]
            self.rl_buffer.append(
                {
                    "frames": frames,
                    "actions": actions,
                    "obj_acs": obj_acs,
                    "rewards": rewards,
                    "terminals": terminals,
                    "language": language,
                }
            )

    def get_data_from_RL_buffer(self, i):
        traj_dict = self.rl_buffer[i]
        skill_dict = {}
        if len(traj_dict["actions"]) > self.max_skill_length:
            start = random.randint(0, len(traj_dict["actions"]) - self.max_skill_length)
            skill_dict["skill_feature"] = traj_dict["frames"][
                start : start + self.max_skill_length + 1
            ]
            skill_dict["low_action"] = traj_dict["actions"][
                start : start + self.max_skill_length
            ]
            skill_dict["object_ids"] = traj_dict["obj_acs"][
                start : start + self.max_skill_length
            ]
            skill_dict["valid_interact"] = (
                traj_dict["obj_acs"][start : start + self.max_skill_length] != 0
            )
            skill_dict["reward"] = traj_dict["rewards"][
                start : start + self.max_skill_length
            ]
            skill_dict["terminal"] = traj_dict["terminals"][
                start : start + self.max_skill_length
            ]

        else:
            skill_dict["skill_feature"] = traj_dict["frames"]
            skill_dict["low_action"] = traj_dict["actions"]
            skill_dict["object_ids"] = traj_dict["obj_acs"]
            skill_dict["valid_interact"] = traj_dict["obj_acs"] != 0
            skill_dict["reward"] = traj_dict["rewards"]
            skill_dict["terminal"] = traj_dict["terminals"]
        skill_dict["ann_token"] = traj_dict["language"]
        skill_dict["token_length"] = traj_dict["language"].shape[0]  # token length
        skill_dict["feature_length"] = skill_dict["skill_feature"].shape[
            0
        ]  # feature length one more than low action number
        skill_dict["annotation"] = traj_dict["language"]  # for compatibility
        return skill_dict

    def update_skill_to_completion_map(self, lang_instr, dones):
        # append if we successfully completed the skill or not so iql_sampler can figure out whether to embed the new skill or primitive skills
        # as policy goals
        if lang_instr not in self.skill_to_completion_map:
            self.skill_to_completion_map[lang_instr] = []
        self.skill_to_completion_map[lang_instr].append(dones[-1].item())


# saving the updated vocab in case there are new words
if __name__ == "__main__":
    path = "../px_13b_next_skill/px_alfred_data_feat_obj-id_merge_goto_composite_opt-13b_add_next_skill"
    dataset = ETIQLFeeder(path, "train", True, 5)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_func,
        # pin_memory=True,
    )
    for item in tqdm(dataloader):

        pass
    torch.save(dataset.vocab_ann, "ET/human.vocab")
