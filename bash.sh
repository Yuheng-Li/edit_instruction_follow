

# # - - - - - - - - - - - - - - For evalsample.json (fake images) eval - - - - - - - - - - - - - - - - - # 

# CUDA_VISIBLE_DEVICES=0 python llava_score_evaluator.py \
#     --enable_mask \
#     --llava_model_path ../edit_instruction_follow_ckpt/llava-v1.5-13b-init_try_700k_mask/ \
#     --test_json_file ../edit_instruction_follow_data/evalsample.json \
#     --test_images_folder s3://myedit-cz/upsample_1k/_nomask/masked_results/stock5M_ediffiN130_v1/merge_three_segs/ \
#     --test_masks_folder s3://myedit-cz/merge_three_segs/stock5M_ediffiN130_v1/ \
#     --total_chunk 8 \
#     --chunk_idx 0 \
#     --output_jsonl_file_path result.jsonl  & 




CUDA_VISIBLE_DEVICES=0 python llava_score_evaluator.py \
    --enable_mask \
    --llava_model_path ../edit_instruction_follow_ckpt/llava-v1.5-13b-init_try_700k_mask/ \
    --test_json_file ../edit_instruction_follow_data/evalsample_real_img.json \
    --test_images_folder ../edit_instruction_follow_data/evalsample_real_img \
    --test_masks_folder s3://myedit-cz/stock_grounding/stock_realimg/bg_7-5_gd_1-5_dilate10/ \
    --output_jsonl_file_path result_real.jsonl  \
    --total_chunk 8 \
    --chunk_idx 0 & 


CUDA_VISIBLE_DEVICES=1 python llava_score_evaluator.py \
    --enable_mask \
    --llava_model_path ../edit_instruction_follow_ckpt/llava-v1.5-13b-init_try_700k_mask/ \
    --test_json_file ../edit_instruction_follow_data/evalsample_real_img.json \
    --test_images_folder ../edit_instruction_follow_data/evalsample_real_img \
    --test_masks_folder s3://myedit-cz/stock_grounding/stock_realimg/bg_7-5_gd_1-5_dilate10/ \
    --output_jsonl_file_path result_real.jsonl  \
    --total_chunk 8 \
    --chunk_idx 1 & 


CUDA_VISIBLE_DEVICES=2 python llava_score_evaluator.py \
    --enable_mask \
    --llava_model_path ../edit_instruction_follow_ckpt/llava-v1.5-13b-init_try_700k_mask/ \
    --test_json_file ../edit_instruction_follow_data/evalsample_real_img.json \
    --test_images_folder ../edit_instruction_follow_data/evalsample_real_img \
    --test_masks_folder s3://myedit-cz/stock_grounding/stock_realimg/bg_7-5_gd_1-5_dilate10/ \
    --output_jsonl_file_path result_real.jsonl  \
    --total_chunk 8 \
    --chunk_idx 2 & 



CUDA_VISIBLE_DEVICES=3 python llava_score_evaluator.py \
    --enable_mask \
    --llava_model_path ../edit_instruction_follow_ckpt/llava-v1.5-13b-init_try_700k_mask/ \
    --test_json_file ../edit_instruction_follow_data/evalsample_real_img.json \
    --test_images_folder ../edit_instruction_follow_data/evalsample_real_img \
    --test_masks_folder s3://myedit-cz/stock_grounding/stock_realimg/bg_7-5_gd_1-5_dilate10/ \
    --output_jsonl_file_path result_real.jsonl  \
    --total_chunk 8 \
    --chunk_idx 3 & 











CUDA_VISIBLE_DEVICES=4 python llava_score_evaluator.py \
    --enable_mask \
    --llava_model_path ../edit_instruction_follow_ckpt/llava-v1.5-13b-init_try_700k_mask/ \
    --test_json_file ../edit_instruction_follow_data/evalsample_real_img.json \
    --test_images_folder ../edit_instruction_follow_data/evalsample_real_img \
    --test_masks_folder s3://myedit-cz/stock_grounding/stock_realimg/bg_7-5_gd_1-5_dilate10/ \
    --output_jsonl_file_path result_real.jsonl  \
    --total_chunk 8 \
    --chunk_idx 4 & 


CUDA_VISIBLE_DEVICES=5 python llava_score_evaluator.py \
    --enable_mask \
    --llava_model_path ../edit_instruction_follow_ckpt/llava-v1.5-13b-init_try_700k_mask/ \
    --test_json_file ../edit_instruction_follow_data/evalsample_real_img.json \
    --test_images_folder ../edit_instruction_follow_data/evalsample_real_img \
    --test_masks_folder s3://myedit-cz/stock_grounding/stock_realimg/bg_7-5_gd_1-5_dilate10/ \
    --output_jsonl_file_path result_real.jsonl  \
    --total_chunk 8 \
    --chunk_idx 5 & 


CUDA_VISIBLE_DEVICES=6 python llava_score_evaluator.py \
    --enable_mask \
    --llava_model_path ../edit_instruction_follow_ckpt/llava-v1.5-13b-init_try_700k_mask/ \
    --test_json_file ../edit_instruction_follow_data/evalsample_real_img.json \
    --test_images_folder ../edit_instruction_follow_data/evalsample_real_img \
    --test_masks_folder s3://myedit-cz/stock_grounding/stock_realimg/bg_7-5_gd_1-5_dilate10/ \
    --output_jsonl_file_path result_real.jsonl  \
    --total_chunk 8 \
    --chunk_idx 6 & 



CUDA_VISIBLE_DEVICES=7 python llava_score_evaluator.py \
    --enable_mask \
    --llava_model_path ../edit_instruction_follow_ckpt/llava-v1.5-13b-init_try_700k_mask/ \
    --test_json_file ../edit_instruction_follow_data/evalsample_real_img.json \
    --test_images_folder ../edit_instruction_follow_data/evalsample_real_img \
    --test_masks_folder s3://myedit-cz/stock_grounding/stock_realimg/bg_7-5_gd_1-5_dilate10/ \
    --output_jsonl_file_path result_real.jsonl  \
    --total_chunk 8 \
    --chunk_idx 7 & 