# inside the container
horovodrun -np 4 python src/tasks/run_video_qa_star.py \
  --config src/configs/star.json \
  --do_inference 1 --output_dir storage/finetune/star\
  --inference_model_step 8120 \
  --inference_batch_size 32 \
  --inference_n_clips 1 \
  --inference_img_db storage/video_db/tokens_star_test \
  --inference_txt_db storage/txt_db/test.jsonl