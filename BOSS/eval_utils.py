import torch
import numpy as np
import gen.constants as constants
import random
import re
import textacy
import spacy

nlp = spacy.load("en_core_web_sm")


visibility_distance = constants.VISIBILITY_DISTANCE

interactive_actions = [
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
]
knives = ["ButterKnife", "Knife"]


# inventory_object_id = self.last_event.metadata['inventoryObjects'][0]['objectId']


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
        invalid_action_mask.append(vocab["action_low"].word2index("PickupObject") - 3)
    if len(openable_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("OpenObject") - 3)
    if (
        len(sliceable_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) == 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("SliceObject") - 3)
    if len(closeable_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("CloseObject") - 3)
    if (
        len(receptacle_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) == 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("PutObject") - 3)
    if len(toggleon_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("ToggleObjectOn") - 3)
    if len(toggleoff_objects) == 0:
        invalid_action_mask.append(
            vocab["action_low"].word2index("ToggleObjectOff") - 3
        )
    if (
        len(env.last_event.metadata["inventoryObjects"]) > 0
        and env.last_event.metadata["inventoryObjects"][0]["objectId"].split("|")[0]
        not in knives
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("SliceObject") - 3)
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


def make_primitive_annotation_eval_dataset(eval_list: list[dict]):
    """
    Make a dataset for evaluation of primitive annotations.
    """
    new_eval_dataset = []
    for eval_dict in eval_list:
        eval_dict_copy = eval_dict.copy()
        annotations = []
        for primitive_skill in eval_dict_copy["primitive_skills"]:
            annotations.append(primitive_skill["annotations"])
        annotations = process_skill_strings(annotations)
        eval_dict_copy["annotation"] = " ".join(annotations)
        new_eval_dataset.append(eval_dict_copy)
    return new_eval_dataset


def add_prefix_to_skill_string(strings: list):
    prefix_skill_list = []
    for skill in strings:
        # get svo tuples
        # svo_tuples = findSVOs(nlp(process_skill_strings([f"I {generation}"])[0]))
        svo_tuples = textacy.extract.subject_verb_object_triples(nlp(f"I {skill}"))
        # turn it back into a sentence without the added I
        svo_tuples = list(svo_tuples)
        if len(svo_tuples) > 0:
            prefix_skill_list.append(f"{svo_tuples[0].verb[0].text.upper()}: {skill}")
        else:
            prefix_skill_list.append(skill)
    return prefix_skill_list


def process_skill_strings(strings):
    processed_strings = []
    for string in strings:
        if isinstance(string, list):
            # artificact of bug in the data
            string = string[0]
        string = string.strip().lower()
        string = re.sub(" +", " ", string)  # remove extra spaces
        if len(string) > 0 and string[-1] not in ["?", ".", "!"]:
            string = string + "."
        processed_strings.append(string.capitalize())
    return processed_strings


def compute_distance(agent_position, object):

    agent_location = np.array(
        [agent_position["x"], agent_position["y"], agent_position["z"]]
    )
    object_location = np.array(
        [object["position"]["x"], object["position"]["y"], object["position"]["z"]]
    )

    distance = np.linalg.norm(agent_location - object_location)

    return distance


def compute_visibility_based_on_distance(agent_position, object, visibility_distance):
    # directly copied from C++ code here https://github.com/allenai/ai2thor/blob/f39ae981646d689047ba7006cb9c1dc507a247ff/unity/Assets/Scripts/BaseFPSAgentController.cs#L2628
    is_visible = True
    x_delta = object["position"]["x"] - agent_position["x"]
    y_delta = object["position"]["y"] - agent_position["y"]
    z_delta = object["position"]["z"] - agent_position["z"]
    if abs(x_delta) > visibility_distance:
        is_visible = False
    elif abs(y_delta) > visibility_distance:
        is_visible = False
    elif abs(z_delta) > visibility_distance:
        is_visible = False
    elif (
        x_delta * x_delta + z_delta * z_delta
        > visibility_distance * visibility_distance
    ):
        is_visible = False
    return is_visible


def forward_model(model, feat, vocab, policy_sample):
    # model = model.to(torch.device('cuda'))
    frames_buffer = feat["frames_buffer"]
    language_latent = feat["language_latent"]
    instance_mask = feat["instance_mask"]

    action_out, output_mask, progress_out = model(
        frames_buffer, language_latent, instance_mask
    )
    if policy_sample == "argmax":
        action_out = torch.argmax(action_out)

    else:
        action_out = action_out.squeeze(1)
        action_out = torch.softmax(action_out, dim=1)
        action_out = torch.multinomial(action_out, 1)

    use_index = action_out + 3
    output_mask = torch.sigmoid(output_mask)
    output_mask = torch.round(output_mask).cpu().detach().numpy()

    string_act = vocab["action_low"].index2word(use_index)

    if string_act not in interactive_actions:
        output_mask = None
    else:
        output_mask = np.squeeze(output_mask, axis=0)
        # ppp_mask = Image.fromarray(np.uint8(output_mask)*255)
        # ppp_mask.save("ins_map_pred.jpg")

    return string_act, output_mask, progress_out


def forward_model_object_id(model, feat, vocab, vocab_obj, policy_sample, env):
    # model = model.to(torch.device('cuda'))
    frames_buffer = feat["frames_buffer"]
    language_latent = feat["language_latent"]
    # instance_mask = feat["instance_mask"]
    instance_mask = None

    action_out, _, object_pred_id, progress_out = model(
        frames_buffer, language_latent, instance_mask
    )

    if policy_sample == "argmax":
        action_out = torch.argmax(action_out)

    else:
        action_out = action_out.squeeze(1)
        action_out = torch.softmax(action_out, dim=1)
        action_out = torch.multinomial(action_out, 1)

    use_index = action_out + 3

    string_act = vocab["action_low"].index2word(use_index)

    if string_act not in interactive_actions:

        return string_act, None, progress_out

    else:

        object_pred_prob = object_pred_id.squeeze(1)
        object_pred_prob = (
            torch.softmax(object_pred_prob, dim=1).squeeze(0).cpu().detach().numpy()
        )

        # choose largest prob visible object
        visible_object = [
            obj for obj in env.last_event.metadata["objects"] if obj["visible"]
        ]

        # print("visible_object",visible_object)

        vis_obj_type = [vis_obj["objectId"].split("|")[0] for vis_obj in visible_object]

        vis_obj_type_id = [
            vocab_obj.word2index(vis_obj_type_use)
            for vis_obj_type_use in vis_obj_type
            if vis_obj_type_use in vocab_obj.to_dict()["index2word"]
        ]
        vis_obj_type_id = list(set(vis_obj_type_id))

        prob_dict = {}
        for id in vis_obj_type_id:
            prob_dict[id] = object_pred_prob[id]
        prob_value = prob_dict.values()
        max_prob = max(prob_value)
        choose_id = [k for k, v in prob_dict.items() if v == max_prob][0]

        # choose the closest object

        object_type = vocab_obj.index2word(choose_id)
        candidate_objects = [
            obj
            for obj in visible_object
            if obj["objectId"].split("|")[0] == object_type
        ]
        # object type
        #
        # print("candidate_objects",candidate_objects)
        if len(candidate_objects) == 1:
            output_object = candidate_objects[0]["objectId"]
        else:

            agent_position = env.last_event.metadata["agent"]["position"]

            min_distance = 10e6

            for ava_object in candidate_objects:

                obj_agent_dist = compute_distance(agent_position, ava_object)
                if obj_agent_dist < min_distance:
                    min_distance = obj_agent_dist
                    output_object = ava_object["objectId"]

        return string_act, output_object, progress_out


def forward_model_object_id_ability_choice(
    model, feat, vocab, vocab_obj, policy_sample, env
):
    # model = model.to(torch.device('cuda'))
    frames_buffer = feat["frames_buffer"]
    language_latent = feat["language_latent"]
    # instance_mask = feat["instance_mask"]
    instance_mask = None

    action_out, _, object_pred_id, progress_out = model(
        frames_buffer, language_latent, instance_mask
    )

    if policy_sample == "argmax":
        action_out_use = torch.argmax(action_out)

    else:
        action_out_use = action_out_use.squeeze(1)
        action_out_use = torch.softmax(action_out_use, dim=1)
        action_out_use = torch.multinomial(action_out_use, 1)

    use_index = action_out_use + 3

    string_act = vocab["action_low"].index2word(use_index)

    if string_act == "PutObject":
        if len(env.last_event.metadata["inventoryObjects"][0]["objectId"]) == 0:
            action_out[0, action_out_use] = 0
            action_out_use = torch.argmax(action_out)
            use_index = action_out_use + 3
            string_act = vocab["action_low"].index2word(use_index)

    if string_act not in interactive_actions:

        return string_act, None, progress_out

    else:

        object_pred_prob = object_pred_id.squeeze(1)
        object_pred_prob = (
            torch.softmax(object_pred_prob, dim=1).squeeze(0).cpu().detach().numpy()
        )

        # choose largest prob visible object
        visible_object = [
            obj for obj in env.last_event.metadata["objects"] if obj["visible"]
        ]
        # interactive_actions = ['PickupObject', 'PutObject', 'OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject']

        if string_act == "PickupObject":
            candidate_objects = [
                obj for obj in visible_object if obj["pickupable"] == True
            ]
        elif string_act == "OpenObject":
            candidate_objects = [
                obj for obj in visible_object if obj["openable"] == True
            ]
        elif string_act == "SliceObject":
            candidate_objects = [
                obj for obj in visible_object if obj["sliceable"] == True
            ]
        elif string_act == "CloseObject":
            candidate_objects = [obj for obj in visible_object if obj["isOpen"] == True]
        elif string_act == "PutObject":
            candidate_objects = [
                obj for obj in visible_object if obj["receptacle"] == True
            ]
        elif string_act == "ToggleObjectOn":
            candidate_objects = [
                obj for obj in visible_object if obj["toggleable"] == True
            ]
        elif string_act == "ToggleObjectOff":
            candidate_objects = [
                obj for obj in visible_object if obj["toggleable"] == True
            ]

        candidate_obj_type = [
            vis_obj["objectId"].split("|")[0] for vis_obj in candidate_objects
        ]

        candidate_obj_type_id = [
            vocab_obj.word2index(candidate_obj_type_use)
            for candidate_obj_type_use in candidate_obj_type
            if candidate_obj_type_use in vocab_obj.to_dict()["index2word"]
        ]
        candidate_obj_type_id = list(set(candidate_obj_type_id))

        if len(candidate_obj_type_id) > 0:
            prob_dict = {}
            for id in candidate_obj_type_id:
                prob_dict[id] = object_pred_prob[id]
            prob_value = prob_dict.values()
            max_prob = max(prob_value)
            choose_id = [k for k, v in prob_dict.items() if v == max_prob][0]

            # choose the closest object

            object_type = vocab_obj.index2word(choose_id)
            candidate_objects = [
                obj
                for obj in candidate_objects
                if obj["objectId"].split("|")[0] == object_type
            ]
            # object type
            #
            # print("candidate_objects",candidate_objects)
            if len(candidate_objects) == 1:
                output_object = candidate_objects[0]["objectId"]
            else:

                agent_position = env.last_event.metadata["agent"]["position"]

                min_distance = 10e6

                for ava_object in candidate_objects:

                    obj_agent_dist = compute_distance(agent_position, ava_object)
                    if obj_agent_dist < min_distance:
                        min_distance = obj_agent_dist
                        output_object = ava_object["objectId"]

        else:
            # choose highest prob visible object
            candidate_obj_type = [
                vis_obj["objectId"].split("|")[0] for vis_obj in visible_object
            ]
            candidate_obj_type_id = [
                vocab_obj.word2index(candidate_obj_type_use)
                for candidate_obj_type_use in candidate_obj_type
                if candidate_obj_type_use in vocab_obj.to_dict()["index2word"]
            ]
            candidate_obj_type_id = list(set(candidate_obj_type_id))
            prob_dict = {}
            for id in candidate_obj_type_id:
                prob_dict[id] = object_pred_prob[id]
            prob_value = prob_dict.values()
            max_prob = max(prob_value)
            choose_id = [k for k, v in prob_dict.items() if v == max_prob][0]

            # choose the closest object

            object_type = vocab_obj.index2word(choose_id)
            candidate_objects = [
                obj
                for obj in visible_object
                if obj["objectId"].split("|")[0] == object_type
            ]

            if len(candidate_objects) == 1:
                output_object = candidate_objects[0]["objectId"]
            else:

                agent_position = env.last_event.metadata["agent"]["position"]

                min_distance = 10e6

                for ava_object in candidate_objects:

                    obj_agent_dist = compute_distance(agent_position, ava_object)
                    if obj_agent_dist < min_distance:
                        min_distance = obj_agent_dist
                        output_object = ava_object["objectId"]

        return string_act, output_object, progress_out


def mask_and_resample(action_probs, action_mask, deterministic, take_rand_action):
    action_probs[0, action_mask] = 0
    if torch.all(action_probs[0] == 0):
        # set the indicies NOT in action mask to 0
        action_mask_complement = np.ones(action_probs.shape[1], dtype=bool)
        action_mask_complement[action_mask] = False
        action_probs[0, action_mask_complement] = 1
    logprobs = torch.log(action_probs)
    # in case all probabilities are 0
    if deterministic:
        chosen_action = torch.argmax(action_probs)
    else:
        dist = torch.distributions.Categorical(logits=logprobs)
        chosen_action = dist.sample()
    if take_rand_action:
        action_mask_complement = np.ones(action_probs.shape[1], dtype=bool)
        # anything that doesn't get masked out by action_mask is in action_mask_complement
        action_mask_complement[action_mask] = False
        # set uniform probability for all valid actions
        # logprobs[0, action_mask_complement] = 0
        action_probs[0, action_mask_complement] = 1
        # sample uniformly
        dist = torch.distributions.Categorical(action_probs, validate_args=False)
        chosen_action = dist.sample()
    return chosen_action


def forward_model_object_id_ability_choice_iql(
    model, feat, vocab, vocab_obj, env, deterministic, epsilon,
):
    take_rand_action = random.random() < epsilon

    frames_buffer = feat["frames_buffer"]
    language_latent = feat["language_latent"]

    action_out, object_pred_id, progress_out, value = model.get_action_dists(
        frames_buffer, language_latent, ret_value_func=False,
    )

    action_out = action_out.probs
    action_out = torch.softmax(action_out, dim=1)

    object_pred_id = object_pred_id.probs
    object_pred_prob = torch.softmax(object_pred_id, dim=1)

    agent_position = env.last_event.metadata["agent"]["position"]

    visible_object = [
        obj
        for obj in env.last_event.metadata["objects"]
        if (
            obj["visible"] == True
            and compute_visibility_based_on_distance(
                agent_position, obj, visibility_distance
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
    string_act = vocab["action_low"].index2word(chosen_action + 3)
    if string_act not in interactive_actions:
        return string_act, None, progress_out, value
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
        choose_id = torch.multinomial(
            torch.tensor(list(prob_value), dtype=torch.float32), 1
        )[0].item()
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
    return string_act, output_object, progress_out, value


def forward_model_object_id_ability_choice_iql_random_action(
    model, feat, vocab, vocab_obj, env, deterministic, epsilon,
):
    take_rand_action = random.random() < epsilon

    frames_buffer = feat["frames_buffer"]
    language_latent = feat["language_latent"]

    action_out, object_pred_id, progress_out, value = model.get_action_dists(
        frames_buffer, language_latent, ret_value_func=True,
    )
    # progress_out = torch.tensor(0)
    # value = torch.tensor(0)
    # action_out = action_out.probs
    action_out = torch.zeros([1, 12])
    for i in range(12):
        action_out[0, i] = 1 / 12

    action_out = torch.softmax(action_out, dim=1)

    object_pred_id = object_pred_id.probs
    # object_pred_id = torch.zeros([1,82])
    # for i in range (82):
    #     object_pred_id[0,i] = 1/82
    object_pred_prob = torch.softmax(object_pred_id, dim=1)

    agent_position = env.last_event.metadata["agent"]["position"]

    visible_object = [
        obj
        for obj in env.last_event.metadata["objects"]
        if (
            obj["visible"]
            and compute_distance(agent_position, obj) <= visibility_distance
        )
    ]
    invalid_action_mask, ret_dict = generate_invalid_action_mask_and_objects(
        env, visible_object, vocab_obj, vocab
    )
    # choose the action after filtering with the mask
    chosen_action = mask_and_resample(
        action_out, invalid_action_mask, deterministic, take_rand_action
    )
    string_act = vocab["action_low"].index2word(chosen_action + 3)
    if string_act not in interactive_actions:
        return string_act, None, progress_out, value
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
        choose_id = torch.multinomial(torch.tensor(list(prob_value)), 1)[0].item()
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
    return string_act, output_object, progress_out, value
