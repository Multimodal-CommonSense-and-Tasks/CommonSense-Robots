import os
import pickle
from custom_lmdb_reader import CustomLMDBReader
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import revtok
from torch.nn.utils.rnn import pad_sequence
import random


def load_object_class(vocab_obj, object_name):
    """
    load object classes for interactive actions
    """
    if object_name is None:
        return 0
    object_class = object_name.split("|")[0]
    return vocab_obj.word2index(object_class)


def remove_spaces(s):
    cs = " ".join(s.split())
    return cs


def remove_spaces_and_lower(s):
    cs = remove_spaces(s)
    cs = cs.lower()
    return cs


def numericalize(vocab, words, train=True):
    """
    converts words to unique integers
    """
    if not train:
        new_words = set(words) - set(vocab["word"].counts.keys())
        if new_words:
            # replace unknown words with <<pad>>
            words = [w if w not in new_words else "<<pad>>" for w in words]
    before = len(vocab["word"])
    ret = vocab["word"].word2index(words, train=train)
    after = len(vocab["word"])
    if after > before:
        print(before, after, words)
    return ret
    # return vocab["word"].word2index(words, train=train)


def load_pyxis(path):
    df = CustomLMDBReader(path, lock=False)
    return df


class ETCompositeFeeder(Dataset):
    def __init__(
        self, path, data_type, max_skill_length,
    ):

        self.path = path
        self.data = self.load_pyxis()
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

        pkl_name = "ET/ET_composite_skill_set_" + data_type

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

        for i in tqdm(range(len(self.data))):

            num_skill = self.data[i]["lang_ridx"].shape[0]

            for j in range(num_skill):
                single_sample_trajectory_dict[total_samples] = (i, j)
                total_samples += 1

        self.single_sample_trajectory_dict = single_sample_trajectory_dict

    def load_pyxis(self):
        df = CustomLMDBReader(self.path, lock=False)
        return df

    def __len__(self):
        return len(self.single_sample_trajectory_dict)

    def __getitem__(self, idx):
        i, j = self.single_sample_trajectory_dict[idx]
        return self.get_data_from_pyxis(i, j)

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

        num_skill = len(data_dict["skill_switch_point"])
        start_index = data_dict["skill_switch_point"][skill_start_index]

        if skill_end_index == len(data_dict["skill_switch_point"]) - 1:
            skill_feature = data_dict["traj_resnet_feature"][start_index:]

        else:
            end_index = (
                data_dict["skill_switch_point"][skill_end_index + 1] + 1
            )  # goal state include the last state of skill combination
            skill_feature = data_dict["traj_resnet_feature"][start_index:end_index]
        skill_feature = torch.from_numpy(skill_feature).float()[
            :-1
        ]  # (s, a) pairs, don't need last state
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

        composite_skill_progress = (
            torch.arange(low_action.shape[0]) + 1
        ) / low_action.shape[0]

        primitive_skill_progress = torch.zeros(low_action.shape[0])

        id_start = data_dict["skill_switch_point"][subgoal_idx[0]]
        last_end = 0
        for kk in subgoal_idx:

            if kk == len(data_dict["skill_switch_point"]) - 1:
                start = last_end
                end = len(low_action)
                id_start = 0
            else:
                start = data_dict["skill_switch_point"][kk]
                end = data_dict["skill_switch_point"][kk + 1]

            real_end = end - start
            real_start = 1

            progress = torch.arange(real_start, real_end + 1) / (
                real_end - real_start + 1
            )
            primitive_skill_progress[start - id_start : end - id_start] = progress
            last_end = end - id_start

        start = random.randint(0, len(low_action) - 1)
        skill_dict["skill_feature"] = skill_feature[
            start : start + self.max_skill_length
        ]  # shape:  batch x seq_len x 512 x 7 x 7
        skill_dict["low_action"] = (
            low_action[start : start + self.max_skill_length] - 1
        )  # shape: batch x action_len
        skill_dict["object_ids"] = object_ids[
            start : start + self.max_skill_length
        ]  # shape: batch x object
        skill_dict["valid_interact"] = valid_interact[
            start : start + self.max_skill_length
        ]
        skill_dict["annotation"] = annotation
        skill_dict["ann_token"] = ann_token
        skill_dict["composite_skill_progress"] = composite_skill_progress[
            start : start + self.max_skill_length
        ]
        skill_dict["primitive_skill_progress"] = primitive_skill_progress[
            start : start + self.max_skill_length
        ]
        skill_dict["token_length"] = ann_token.shape[0]  # token length
        skill_dict["skill_length"] = skill_dict["low_action"].shape[0]
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
    composite_skill_progress = []
    skill_length = []
    feature_length = []
    token_length = []
    primitive_skill_progress = []
    for i in range(batch_len):
        dic = batch_dic[i]
        skill_feature.append(dic["skill_feature"])
        low_action.append(dic["low_action"])
        object_ids.append(dic["object_ids"])
        valid_interact.append(dic["valid_interact"])
        ann_token.append(dic["ann_token"])
        composite_skill_progress.append(dic["composite_skill_progress"])
        primitive_skill_progress.append(dic["primitive_skill_progress"])
        skill_length.append(dic["skill_length"])
        feature_length.append(dic["feature_length"])
        token_length.append(dic["token_length"])
        annotations.append(dic["annotation"])

    res = {}
    res["skill_feature"] = pad_sequence(
        skill_feature, batch_first=True, padding_value=0
    )
    res["low_action"] = pad_sequence(low_action, batch_first=True, padding_value=0)
    res["object_ids"] = pad_sequence(object_ids, batch_first=True, padding_value=0)
    res["valid_interact"] = pad_sequence(
        valid_interact, batch_first=True, padding_value=0
    )
    res["ann_token"] = pad_sequence(ann_token, batch_first=True, padding_value=0)
    res["composite_skill_progress"] = pad_sequence(
        composite_skill_progress, batch_first=True, padding_value=0
    )
    res["primitive_skill_progress"] = pad_sequence(
        primitive_skill_progress, batch_first=True, padding_value=0
    )
    res["skill_length"] = torch.tensor(np.asarray(skill_length))
    res["feature_length"] = torch.tensor(np.asarray(feature_length))
    res["token_length"] = torch.tensor(np.asarray(token_length))
    res["annotation"] = annotations

    return res


# saving the updated vocab in case there are new words
if __name__ == "__main__":
    path = "/data/jzhang96/px_13b_next_skill/px_alfred_data_feat_obj-id_merge_goto_composite_opt-13b_add_next_skill"
    dataset = ETCompositeFeeder(path, "train", 5)
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
