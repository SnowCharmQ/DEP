import os
import sys
import torch
import warnings
import torch.distributed as dist

from datasets import load_from_disk
from transformers import set_seed
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from model.personal_model import DEPModel

warnings.filterwarnings("ignore")

set_seed(42)

class CustomTrainer(Seq2SeqTrainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"output/checkpoint-{self.state.global_step}"
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        self.save_model(checkpoint_folder)
        self.tokenizer.save_pretrained(checkpoint_folder)
        print(f"Checkpoint saved to {checkpoint_folder}")

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

llm_model_name = "Qwen/Qwen2.5-7B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained("output/DEP-tokenizer")
personal_model = DEPModel.from_pretrained(
    llm_model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation='flash_attention_2',
    training=True,
    tokenizer=llm_tokenizer,
)
personal_model.resize_token_embeddings(len(llm_tokenizer), mean_resizing=False)

print(personal_model)
print_trainable_parameters(personal_model)
training_args = Seq2SeqTrainingArguments(
    num_train_epochs=5,
    output_dir=f"output",
    logging_steps=10,
    save_strategy="epoch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="adamw_torch",
    learning_rate=1e-5,
    weight_decay=0.025,
    warmup_ratio=0.01,
    bf16=True,
    deepspeed="deepspeed/ds_z1_config.json",
    report_to="wandb",
    run_name="DEP",
)

personal_dataset = load_from_disk("data/dataset_train")
trainer = CustomTrainer(
    model=personal_model,
    args=training_args,
    train_dataset=personal_dataset,
    tokenizer=llm_tokenizer,
)

print("train start")
trainer.train()
print("train done")
if dist.is_initialized():
    dist.destroy_process_group()
sys.exit(0)
