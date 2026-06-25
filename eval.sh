```bash
#!/bin/bash

# Uncomment the section you want to run.

################################################################################
# Zero-shot LLMs
################################################################################

python main.py \
    --mode=inference \
    --model_type=llama \
    --model_name=meta-llama/Llama-3.1-8B-Instruct \
    --testset=gossipcop_test_sen_llama_neg \
    --use_finetuned_model=False

# Example for Qwen:
# python main.py \
#     --mode=inference \
#     --model_type=qwen \
#     --model_name=Qwen/Qwen2.5-7B-Instruct \
#     --testset=gossipcop_test_sen_llama_neg \
#     --use_finetuned_model=False

################################################################################
# Training AdSent
# AdSent is trained by fine-tuning LLaMA-3.1-8B-Instruct on the neutral variant
# of the training set for each dataset.
################################################################################

# python main.py \
#     --mode=train \
#     --model_type=llama \
#     --model_name=meta-llama/Llama-3.1-8B-Instruct \
#     --trainset=politifact_train_sen_llama_neutral \
#     --testset=politifact_test_sen_llama_neutral \
#     --use_finetuned_model=False

################################################################################
# Inference AdSent
# Replace CHECKPOINT_DIR with the path to your trained checkpoint.
################################################################################

# python main.py \
#     --mode=inference \
#     --use_finetuned_model=True \
#     --model_type=llama \
#     --checkpoint_dir=CHECKPOINT_DIR \
#     --testset=politifact_test_sen_llama_neutral
```
