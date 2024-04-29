import json
import os 
import re
import random

def read_metadata(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list


def read_instruction(path):
    lines = []
    with open(path, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines[0] # assume they all the same


def create_one_data_point_for_llava(count, image_file_before, image_file_after, instruction, answer):
    
    # trimming and adjusting whitespace, capitalizing the first letter, and ensuring punctuation is formatted correctly.
    instruction = instruction.strip().strip('.').strip().capitalize() + '.'
    instruction = re.sub(r'\s+,', ', ', instruction)
    instruction = re.sub(r'\s+', ' ', instruction)


    prompt_template = """\
    Instruction: {instruction}
    Before Image: <image>
    After Image: <image>
    Question: Does the "After Image" accurately follow the "Instruction" based on the changes from the "Before Image"? Answer Yes/No directly.
    """

    formatted_prompt = prompt_template.format(instruction=instruction)

    conversations = [
        {
            "from": "human",
            "value": formatted_prompt
        },
        {
            "from": "gpt",
            "value": answer
        }
    ]

    out = dict(
        id = str(count).zfill(8),
        image = [image_file_before, image_file_after],
        conversations = conversations
    )

    return out



# - - - - - - - - - - - - args - - - - - - - - - - - - #  
file_path = '../llava_filter_evalsample/metadata.jsonl'
folder_path = '../llava_filter_evalsample/'
balance_good_bad = False 

out_file = 'editing_instruction_following_data.json'





total_count = 0 

good_sample_conversations = []
bad_sample_conversations = []

raw_data = read_metadata(file_path)
for raw_datum in raw_data:

    # fetch instruction
    instruction = read_instruction( os.path.join(folder_path, raw_datum['index'], 'instruction.txt') )

    # get good and bad samples 
    good_samples = set(tuple(item) for item in raw_datum['good_sample'])
    all_samples = set(tuple(item) for item in raw_datum['all_sample'])
    bad_samples = [list(item) for item in all_samples - good_samples]

    
    for sample in good_samples:
        conv = create_one_data_point_for_llava(
            total_count, 
            os.path.join(raw_datum['index'], sample[0] ), 
            os.path.join(raw_datum['index'], sample[1] ),
            instruction,
            'Yes'  
            )
        good_sample_conversations.append( conv )
        total_count += 1


    for sample in bad_samples:
        conv = create_one_data_point_for_llava(
            total_count, 
            os.path.join(raw_datum['index'], sample[0] ), 
            os.path.join(raw_datum['index'], sample[1] ),
            instruction,
            'No'  
            )
        bad_sample_conversations.append( conv )
        total_count += 1
    


if balance_good_bad:
    min_size = min(len(good_sample_conversations), len(bad_sample_conversations))
    good_sample_conversations = good_sample_conversations[:min_size]
    bad_sample_conversations = bad_sample_conversations[:min_size]

save = good_sample_conversations + bad_sample_conversations


with open(out_file, 'w') as json_file:
    json.dump(save, json_file, indent=4)
