import json
import re
import random

target_dataset = "./scene_sampling/valid_unseen_ann_human.json"


def process_skill_strings(strings):
    processed_strings = []
    for string in strings:
        if isinstance(string, list):
            # artificact of bug in the data
            string = string[0]
        string = string.strip().lower()
        string = re.sub(" +", " ", string)  # remove extra spaces
        if string[-1] not in ["?", ".", "!"]:
            string = string + "."
        processed_strings.append(string.capitalize())
    return processed_strings


with open(target_dataset, "r") as f:
    dataset = json.load(f)

length_types = [2, 3, 4, 5, 6, 7]
random.shuffle(dataset)

for single_data in dataset:
    annotation = single_data["annotation"]
    if type(annotation) != list:
        annotation = [annotation]
    annotation = process_skill_strings(annotation)[0]

    primitive_annotations = []
    primitive_skills = single_data["primitive_skills"]
    if len(primitive_skills) in length_types:
        for skill in primitive_skills:
            primitive_annotations.append(skill["annotations"])
        primitive_annotations = process_skill_strings(primitive_annotations)

        formatted_primitive_annotations = []
        for i, primitive_annotation in enumerate(primitive_annotations):
            formatted_primitive_annotations.append(f"{i+1}: {primitive_annotation}")
        formatted_primitive_annotations = "\n".join(formatted_primitive_annotations)
        # print(
        #    f"\n\nLet's think step-by-step:\n{formatted_primitive_annotations}\nSummary: {annotation}"
        # )
        print(
            f"summary_start += '\n\nLet's think step-by-step:\n{formatted_primitive_annotations}\nSummary: {annotation}"
        )
        length_types.remove(len(primitive_skills))
