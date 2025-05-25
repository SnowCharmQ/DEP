import os
import sys
import json
import torch
import argparse
import evaluate
import warnings
import torch.distributed as dist

from tqdm import tqdm
from transformers import set_seed
from datasets import load_from_disk

from utils.utils import postprocess_output

warnings.filterwarnings("ignore")

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--select", type=str, choices=["infer", "eval"], required=True)
parser.add_argument("--version", type=int)
args = parser.parse_args()

if __name__ == "__main__":

    import vllm

    model_dir = "output"
    subfolders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
    model_dirs = [os.path.join(model_dir, f) for f in subfolders]
    model_dirs = sorted(model_dirs, key=lambda x: int(x.split("-")[-1]))

    val_dataset = load_from_disk("data/dataset_val")
    prompts = val_dataset['inp_str']
    his_diff_embs = val_dataset['his_diff_emb']
    his_diff_embs = [torch.tensor(his_diff_emb)
                          for his_diff_emb in his_diff_embs]
    references = val_dataset['out_str']

    if args.select == "infer":
        model_dir = model_dirs[args.version]
        sampling_params = vllm.SamplingParams(
            max_tokens=2048,
            skip_special_tokens=True,
            temperature=0.8,
            top_p=0.95
        )
        llm = vllm.LLM(
            model_dir,
            tokenizer="data/tokenizer",
            gpu_memory_utilization=0.75,
            max_num_batched_tokens=128,
            max_num_seqs=128,
            enforce_eager=True
        )
        predictions = llm.generate(prompts, 
                                   his_diff_embs=his_diff_embs, 
                                   sampling_params=sampling_params)
        predictions = [prediction.outputs[0].text
                    for prediction in tqdm(predictions, desc="Post-processing", total=len(predictions))]
        with open(f"{model_dir}/predictions_val.txt", "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(pred + "\n---------------------------------\n")
        print(f"Done for {model_dir}")

    elif args.select == "eval":
        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load('rouge')
        meteor_metric = evaluate.load('meteor')

        all_results = []

        for model_dir in model_dirs:
            predictions_path = f"{model_dir}/predictions_val.txt"
            with open(predictions_path, 'r', encoding='utf-8') as f:
                predictions = f.read()
                predictions = predictions.split('\n---------------------------------\n')
                predictions = predictions[:-1]
            predictions = [postprocess_output(prediction) for prediction in predictions]
            try:
                result_bleu = bleu_metric.compute(predictions=predictions,
                                                references=references)
                result_rouge = rouge_metric.compute(predictions=predictions,
                                                    references=references)
                result_meteor = meteor_metric.compute(predictions=predictions,
                                                    references=references)
                result = {
                    "model": model_dir,
                    "rouge-1": result_rouge["rouge1"],
                    "rouge-L": result_rouge["rougeL"],
                    "bleu": result_bleu["score"],
                    "meteor": result_meteor['meteor'],
                }
            except Exception as e:
                result = {
                    "model": model_dir,
                    "rouge-1": 0,
                    "rouge-L": 0,
                    "bleu": 0,
                    "meteor": 0,
                }
            all_results.append(result)

        print(all_results)

        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)

if dist.is_initialized():
    dist.destroy_process_group()
sys.exit(0)
