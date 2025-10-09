

accelerate launch train_unconditional.py \
  --train_data_dir="Your_dir" \
  --output_dir="Your_dir" \
  --num_epochs=501 \
  --train_batch_size=128 \
  --eval_batch_size=1024 \
  --save_model_epochs=500 \
  --save_images_epochs=500 \
  --ddpm_num_steps=1000 \
  --ddpm_num_inference_steps=1000 \
  --resolution=32 \
  --model_size="UNet48" \
  --data_size=32768 \
  --syn_type="last_greedyfeature" \
  --mixed_precision="fp16"