import base64
import gzip
import numpy as np
import json
from PIL import Image 
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



def decompressed_descriptor(comb_comp_64_ascii):
    comb_comp_64 = comb_comp_64_ascii.encode()
    comb_comp = base64.b64decode(comb_comp_64)
    comb_bytes = gzip.decompress(comb_comp)
    comb_array_normed = np.frombuffer(comb_bytes, dtype=bool)
    return comb_array_normed

def find_bbox(mask):
    # Find the rows and columns where the mask has foreground pixels
    rows = np.any(mask == 0, axis=1)
    cols = np.any(mask == 0, axis=0)

    # Determine the bounding box coordinates
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    # Return the bounding box in the format (x1, y1, x2, y2)
    return x1, y1, x2, y2

llava_file_path = '../edit_instruction_follow_data/llava_filtering_100k.json'

s3_path = 's3://myedit-cz/upsample_1k/_nomask/masked_results/stock5M_ediffiN130_v1/merge_three_segs/'
s3_mask_path = 's3://myedit-cz/merge_three_segs/stock5M_ediffiN130_v1/'

with open(llava_file_path, 'r') as file:
    data = json.load(file)


random.shuffle(data)
for datum in tqdm(data):
    img0, img1 = datum['image']
    tmp0, tmp1 = img0.split('_')[0].split('/')
    mask_name = tmp0+'_'+tmp1+'_0.json'
    mask_path = os.path.join(s3_mask_path, mask_name)

    mask_info = load_s3_json(mask_path)
    mask = decompressed_descriptor(mask_info['mask'])
    mask = mask.reshape((mask_info['h'], mask_info['w']))

    # save 
    mask = (mask * 255).astype(np.uint8)
    x1, y1, x2, y2 = find_bbox(mask)
    breakpoint()  
    # image = Image.fromarray(mask, mode='L')  
    # image.save('output_mask_image.png')
