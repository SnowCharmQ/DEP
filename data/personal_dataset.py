import torch
import datasets
import numpy as np
from tqdm import tqdm

from utils.templates import Qwen2PromptTemplate

class PersonalDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        main_dataset, 
        meta_dataset, 
        user_his_emb_map,
        user_prof_mean_emb_map,
        asin_reviewers_map,
        llm_tokenizer,
        max_length=3072,
        max_his_len=8,
        new_tokens=None,
        training=True,
    ):
        self.main_dataset = main_dataset
        self.meta_dataset = meta_dataset
        self.llm_tokenizer = llm_tokenizer
        self.total_len = len(self.main_dataset)
        system_prompt = (
            f"Given the title and description of an item, "
            f"along with the user's past reviews (including item title, item description, review rating, review title, review text, review embedding, review difference embedding), "
            f"and the output review rating and review title, "
            f"generate a personalized item review for the user.\n"
            f"Note: [Review Embedding] denotes a soft prompt of the review text and [Review Difference Embedding] denotes a soft prompt showing the difference between the review text and other reviews on the same item. "
            f"[Review Embedding] and [Review Difference Embedding] should serve as hints for personalized review text generation.\n"
        )
        self.pt = Qwen2PromptTemplate(system_prompt)
        self.processed_data = []

        for idx in tqdm(range(self.total_len), desc=f"Pre-Processing data"):
            category = self.main_dataset[idx]["category"]
            data = self.main_dataset[idx]["data"]
            out_str = data["text"]
            item_asin = data["asin"]
            item_title = self.meta_dataset[item_asin][0]
            item_desc = self.meta_dataset[item_asin][1]
            user_id = self.main_dataset[idx]["user_id"]
            profile = self.main_dataset[idx]["profile"]
            prof_len = len(profile)
            profile = sorted(profile, key=lambda x: x["timestamp"], reverse=True)
            user_emb = torch.load(f"embeddings/{category}/{user_id}.emb", weights_only=True)
            user_emb = user_emb[:prof_len]
            user_emb = torch.flip(user_emb, dims=[0])
            for p in profile:
                asin = p["asin"]
                p_item_title, p_item_desc = self.meta_dataset[asin]
                p['item_title'] = p_item_title
                p["item_desc"] = p_item_desc

            his_emb = []
            diff_emb = []
            diff_profile = []
            for jdx, p in enumerate(profile):
                if len(asin_reviewers_map[p["asin"]]) > 1:
                    p_asin = p["asin"]
                    p_user_emb = user_emb[jdx]
                    p_diff_emb = []
                    reviewers = sorted(asin_reviewers_map[p_asin])
                    for r_uid, p_idx in reviewers:
                        if r_uid == user_id:
                            continue
                        p_r_emb = user_his_emb_map[f"{r_uid}_{category}"][p_idx]
                        u_prof_diff = user_prof_mean_emb_map[f"{user_id}_{category}"] 
                        r_prof_diff = user_prof_mean_emb_map[f"{r_uid}_{category}"]
                        prof_diff = u_prof_diff - r_prof_diff
                        p_diff_emb.append((prof_diff, p_user_emb - p_r_emb))                    
                    prof_diffs = torch.stack([item[0] for item in p_diff_emb])
                    weights = torch.norm(prof_diffs, dim=1)
                    weights = weights / torch.sum(weights)
                    weights = torch.softmax(weights, dim=0)
                    p_diff_emb = torch.stack([weights[i] * item[1] 
                                              for i, item in enumerate(p_diff_emb)])
                    p_diff_emb = torch.sum(p_diff_emb, dim=0)
                    his_emb.append(p_user_emb)
                    diff_emb.append(p_diff_emb)
                    diff_profile.append(p)

                if len(his_emb) == len(diff_emb) == len(diff_profile) == 8:
                    break

            his_emb = torch.stack(his_emb)
            diff_emb = torch.stack(diff_emb)
            his_diff_emb = torch.cat([his_emb, diff_emb], dim=0)

            tmp_inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"[Output Review Rating]: {data['rating']}\n"
                f"[Output Review Title]: {data['title']}\n"
            )
            tmp_inp_str = self.pt.build_prompt(tmp_inp_str)
            tmp_ids = self.llm_tokenizer(tmp_inp_str, add_special_tokens=False)['input_ids']
            tmp_len = len(tmp_ids)
            avail_len = max_length - tmp_len
            past_reviews = ""
            for tmp_prof_len in range(max_his_len, 0, -1):
                past_reviews = []
                for i in range(tmp_prof_len):
                    review_info = (
                        f"- [Review {i+1}]:\n"
                        f"  - [Item Title]: {profile[i]['item_title']}\n"
                        f"  - [Item Description]: {profile[i]['item_desc']}\n"
                        f"  - [Review Rating]: {profile[i]['rating']}\n"
                        f"  - [Review Title]: {profile[i]['title']}\n"
                        f"  - [Review Text]: {profile[i]['text']}\n"
                        f"  - [Review Embedding]: <his_token_start>{new_tokens[i]}<his_token_end>\n"
                        f"  - [Review Difference Embedding]: <diff_token_start>{new_tokens[i+8]}<diff_token_end>\n"
                    )
                    past_reviews.append(review_info)
                past_reviews = f"[User's Past Reviews]:\n{''.join(past_reviews)}"
                past_ids = self.llm_tokenizer(past_reviews, add_special_tokens=False)['input_ids']
                if len(past_ids) <= avail_len:
                    break
            inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"{past_reviews}"
                f"[Output Review Rating]: {data['rating']}\n"
                f"[Output Review Title]: {data['title']}\n"
            )
            inp_str = self.pt.build_prompt(inp_str)
            total_max_length = max_length + 2048 + 1
            inputs = self.llm_tokenizer(
                inp_str,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False
            )
            targets = self.llm_tokenizer(
                out_str,
                max_length=max_length, 
                truncation=True,
                add_special_tokens=False
            )
            data = {
                'inp_str': inp_str,
                'out_str': out_str,
                'his_diff_emb': his_diff_emb
            }
            if training:
                inputs_id = inputs['input_ids'] + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                attention_mask = inputs['attention_mask'] + targets['attention_mask'] + [1]
                labels = [-100] * len(inputs['input_ids']) + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                if len(inputs_id) < total_max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (total_max_length - len(inputs_id)) + inputs_id
                    attention_mask = [0] * (total_max_length - len(attention_mask)) + attention_mask
                    labels = [-100] * (total_max_length - len(labels)) + labels
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
                data['attention_mask'] = np.array(attention_mask, dtype=np.int64)
                data['labels'] = np.array(labels, dtype=np.int64)
            else:
                inputs_id = inputs['input_ids']
                if len(inputs_id) < max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (max_length - len(inputs_id)) + inputs_id
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
            self.processed_data.append(data)

    def __len__(self):
        return self.total_len

    def get_output(self, idx):
        return self.main_dataset[idx]["data"]["text"]
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

def convert_to_dataset(dataset):
    def gen():
        for data in dataset:
            yield data
    return datasets.Dataset.from_generator(gen)
