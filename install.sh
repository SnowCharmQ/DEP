conda create -n dep python=3.11.11 -y
conda activate dep

# Evaluation
### It takes a while to install vllm twice, we will try to optimize it in the future.
git clone https://github.com/SnowCharmQ/vllm.git vllm-dep
cd vllm-dep
git checkout dep
pip install vllm==0.7.3
VLLM_USE_PRECOMPILED=1 pip install --editable . # Important!
pip install transformers==4.51.3
pip install evaluate rank_bm25 sacrebleu rouge_score absl-py bert_score
cd ..

# Training
pip install pandas datasets accelerate deepspeed
pip install flash-attn --no-build-isolation
pip install wandb