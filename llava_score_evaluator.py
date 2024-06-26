import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image

import requests
from PIL import Image
# from io import BytesIO
from torchvision import transforms
import json
import os 
from tqdm import tqdm
import random

from flash_s3_dataloader.s3_io import \
    load_s3_image, save_s3_image, \
    load_s3_text, save_s3_text, \
    load_s3_json, save_s3_json, \
    check_s3_exists, list_s3_dir, \
    parallel_upload_folder_to_s3, parallel_download_folder_from_s3, \
    upload_file, download_file, \
    get_s3_filesize, load_s3_exr, \
    save_ckpt_to_s3, load_ckpt_from_s3

class LLaVAEvaluator:
    def __init__(self, model_path=None, model_base=None):

        self.model_path = model_path.rstrip('/')
        self.model_base = model_base
        self.dtype = torch.float16


        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(self.model_path, self.model_base, model_name)
        # self.image_processor_for_tensor = transforms.Compose([
        #     transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] )
        # ])

        if "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt_multimodal"
        else:
            conv_mode = "llava_v0" # it was "multimodal" for the ckpt LLaVA-7B-v0 in old code version
        self.conv_mode = conv_mode



    @torch.no_grad()
    def __call__(self, image0, image1, caption, output_attentions=False):


        # user_question = 'Is this image matched with the caption: '+caption
        # user_question = "Does this image match the following caption: "+ caption + "?\nAnswer Yes/No directly."
        if self.model.config.mm_use_im_start_end:
            raise NotImplementedError
            # user_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_question
        # else:
        #     user_prompt = DEFAULT_IMAGE_TOKEN + '\n' + DEFAULT_IMAGE_TOKEN + '\n'  + user_question
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], caption)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()


        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


        # image tensor
        image0 = load_s3_image(image0).convert('RGB')
        image0_tensor = process_images([image0], self.image_processor, self.model.config).cuda().to(self.dtype)

        image1 = load_s3_image(image1).convert('RGB')
        image1_tensor = process_images([image1], self.image_processor, self.model.config).cuda().to(self.dtype)


        setattr(self.model, 'tokenizer', self.tokenizer) # easy for me to debug
        
        score_info = self.model.get_score(
            input_ids,
            images0=image0_tensor,
            images1=image1_tensor,
            labels=None, 
            use_cache=True,
            score_type='yesno',
            output_attentions=output_attentions,
            )

        return score_info








if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--llava_model_path", type=str, default='liuhaotian/llava-v1.5-13b')
    parser.add_argument("--test_json_file", type=str, default='../edit_instruction_follow_data/evalsample.json', help='')
    parser.add_argument("--test_images_folder", type=str, default='s3://myedit-cz/upsample_1k/_nomask/masked_results/stock5M_ediffiN130_v1/merge_three_segs/')

    parser.add_argument("--output_jsonl_file_path", type=str, default='result.jsonl', help='')
    args = parser.parse_args()


    evaluator = LLaVAEvaluator(model_path=args.llava_model_path, model_base=None)

    list_data_dict = json.load(open(args.test_json_file, "r"))

    # random.shuffle(list_data_dict)
    # list_data_dict =list_data_dict[0:100]


    save = []
    total = len(list_data_dict)
    for data_dict in tqdm(list_data_dict):
        
        image0 = os.path.join(  os.path.join( args.test_images_folder,  data_dict['image'][0]  )  )
        image1 = os.path.join(  os.path.join( args.test_images_folder,  data_dict['image'][1]  )  )

        caption =  data_dict['conversations'][0]['value']
        label = data_dict['conversations'][1]['value']

        with torch.no_grad():
            score_info = evaluator(image0, image1, caption)
            score = score_info['score'].item()

        save.append( dict(label=label, score=score) )


    with open(args.output_jsonl_file_path, 'w') as f:
        for item in save:
            json.dump(item, f)
            f.write('\n')
    # print("Acc: ", (len(results_TP)+len(results_TN)) / total  * 100  )
    # print(" ")
    # print("TOTAL: ", total )
    # print("TP: ", len(results_TP)   )
    # print("TN: ", len(results_TN)   )
    # print("FP: ", len(results_FP)   )
    # print("FN: ", len(results_FN)   )

