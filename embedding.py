import gc
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


emb_model = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(emb_model)
model = AutoModel.from_pretrained(
    emb_model,
    device_map="auto",
)


def process(sample):
    user_id = sample['user_id']
    data = sample['data']
    text = data['text']
    profile = sample['profile']
    profile = sorted(profile, key=lambda x: x["timestamp"], reverse=False)
    prof_inps = [f"{prof['text']}" for prof in profile]
    return user_id, prof_inps, text


@torch.no_grad()
def get_embeddings(all_txts):
    batch_size = 512
    embeddings = []
    for i in range(0, len(all_txts), batch_size):
        batch_txts = all_txts[i:i+batch_size]
        batch_tokens = tokenizer(
            batch_txts, truncation=True, padding=True, return_tensors="pt")
        for key in batch_tokens:
            batch_tokens[key] = batch_tokens[key].to(model.device)
        batch_embeddings = model(**batch_tokens)
        batch_embeddings = torch.nn.functional.normalize(
            batch_embeddings[0][:, 0], p=2, dim=1).detach().cpu()
        embeddings.append(batch_embeddings)
        del batch_tokens, batch_embeddings
        torch.cuda.empty_cache()
        gc.collect()
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


categories = ["Books", "Movies_and_TV", "CDs_and_Vinyl"]
split = "test"
output_dir = "embeddings"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for category in categories:
    os.makedirs(f"{output_dir}/{category}", exist_ok=True)
    main_dataset = load_dataset(
        "SnowCharmQ/DPL-main",
        category,
        split=split
    )
    for sample in tqdm(main_dataset, desc=f"Embedding {category} {split}"):
        user_id, prof_inps, text = process(sample)
        all_txts = prof_inps + [text]
        embeddings = get_embeddings(all_txts)
        torch.save(embeddings, f"{output_dir}/{category}/{user_id}.emb")
