import os
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, set_seed
from datasets import load_dataset, concatenate_datasets, Dataset

from data.personal_dataset import convert_to_dataset, PersonalDataset

set_seed(30)

if not os.path.exists("output"):
    os.makedirs("output")

categories = ["Books", "Movies_and_TV", "CDs_and_Vinyl"]
dataset_len = [317, 1925, 1754]

train_datasets = []
meta_datasets = []
for i, category in enumerate(categories):
    train_dataset = load_dataset(
        "SnowCharmQ/DPL-main",
        category,
        split="train"
    ).map(lambda _: {"category": category})
    df = pd.DataFrame(train_dataset)
    df['profile_length'] = df['profile'].apply(len)
    dataset = (
        df.sort_values('profile_length', ascending=False)
        .groupby('user_id')
        .head(1)
        .reset_index(drop=True)
    )
    train_dataset = Dataset.from_pandas(dataset)
    train_datasets.append(train_dataset)
    meta_dataset = load_dataset(
        "SnowCharmQ/DPL-meta",
        category,
        split="full"
    )
    meta_datasets.append(meta_dataset)

train_dataset = concatenate_datasets(train_datasets)
train_dataset = train_dataset.shuffle(seed=42)
meta_dataset = concatenate_datasets(meta_datasets)
meta_dataset = dict(zip(meta_dataset["asin"],
                        zip(meta_dataset["title"],
                            meta_dataset["description"])))

val_datasets = []
for i, category in enumerate(categories):
    val_main_dataset = load_dataset(
        "SnowCharmQ/DPL-main",
        category,
        split="val"
    ).map(lambda _: {"category": category})
    val_datasets.append(val_main_dataset)
val_dataset = concatenate_datasets(val_datasets)
val_dataset = val_dataset.shuffle(seed=42)
val_dataset = val_dataset.select(range(512))

test_datasets = []
for category in categories:
    test_dataset = load_dataset(
        "SnowCharmQ/DPL-main",
        category,
        split="test"
    ).map(lambda _: {"category": category})
    test_datasets.append(test_dataset)
test_dataset = concatenate_datasets(test_datasets)

llm_model_name = "Qwen/Qwen2.5-7B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_tokenizer.padding_side = "left"
new_tokens = [f"[HIS_TOKEN_{i}]" for i in range(8)] + \
    [f"[DIFF_TOKEN_{i}]" for i in range(8)] + \
    ["<his_token_start>", "<his_token_end>",
        "<diff_token_start>", "<diff_token_end>"]
llm_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
llm_tokenizer.save_pretrained("output/DEP-tokenizer")

user_his_emb_map = {}
user_prof_mean_emb_map = {}
asin_reviewers_map = defaultdict(set)
for sample in tqdm(test_dataset, desc="Pre-Processing the dataset"):
    user_id = sample["user_id"]
    category = sample["category"]
    profile = sample["profile"]
    profile = profile[:-2]
    his_emb = torch.load(f"embeddings/{category}/{user_id}.emb", weights_only=True)
    his_emb = his_emb[:-2]
    prof_mean_emb = torch.mean(his_emb, dim=0)
    user_his_emb_map[f"{user_id}_{category}"] = his_emb
    user_prof_mean_emb_map[f"{user_id}_{category}"] = prof_mean_emb
    for i, p in enumerate(profile):
        asin_reviewers_map[p["asin"]].add((user_id, i))

personal_dataset = PersonalDataset(
    train_dataset,
    meta_dataset,
    user_his_emb_map,
    user_prof_mean_emb_map,
    asin_reviewers_map,
    llm_tokenizer=llm_tokenizer,
    new_tokens=new_tokens,
    training=True
)
hf_dataset = convert_to_dataset(personal_dataset)
hf_dataset.save_to_disk("data/dataset_train")

personal_dataset = PersonalDataset(
    val_dataset,
    meta_dataset,
    user_his_emb_map,
    user_prof_mean_emb_map,
    asin_reviewers_map,
    llm_tokenizer=llm_tokenizer,
    new_tokens=new_tokens,
    training=False
)
hf_dataset = convert_to_dataset(personal_dataset)
hf_dataset.save_to_disk("data/dataset_val")

for i, category in enumerate(categories):
    test_train_dataset = load_dataset(
        "SnowCharmQ/DPL-main",
        category,
        split="test"
    ).map(lambda _: {"category": category})
    test_dataset = PersonalDataset(
        test_train_dataset,
        meta_dataset,
        user_his_emb_map,
        user_prof_mean_emb_map,
        asin_reviewers_map,
        llm_tokenizer=llm_tokenizer,
        new_tokens=new_tokens,
        training=False
    )
    hf_dataset = convert_to_dataset(test_dataset)
    hf_dataset.save_to_disk(f"data/dataset_test_{category}")
