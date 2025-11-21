# Project-Blackberry-MoDE-Training
Package contains the main training methods used for the Mixture-of-Difficulty-Experts model. This model is being trained in conjunction with the supervised finetuning of a llama 3 8B model in order to compare both performance, explainability, and latent signal distributions when finetuning the downstream Large Reasoning Model


MoDE-Train-Init-v10

Init training run to test if orchestration works

# Test Configurations
--learning_rate
5e-4 
--num_epochs
6
--batch_size 
192

# Iterated training runs

## Trial v1
--learning_rate 5e-4 
--num_epochs 50 
--batch_size 192 
--warmup_steps 350 #OPTIONAL
--output_dir gs://your-bucket/your-output-path

## Trial v2
--learning_rate 4e-4 
--num_epochs 200
--batch_size 192 
--output_dir "gs://mode-training-init-us-central-1/followup-run/train-run-v1-200-epochs-with-smaller-warmup"


Desc: Changed up routing to top1 during training, increased num epochs, introduced larger loss weights, lower learning rate
Name: MoDE-Training-Iter-v2

## Trial v3
--learning_rate 
4e-4 
--num_epochs 
200
--batch_size 
192
--gcs_bucket 
mode-training-init-us-central-1
--gcs_path 
followup-run/train-run-v1-200-epochs-with-smaller-warmup