import os
import sys
import json
import torch
import argparse
import evaluate
import warnings
import torch.distributed as dist

from transformers import set_seed
from datasets import load_from_disk

from utils.utils import postprocess_output

warnings.filterwarnings("ignore")

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--dataset", required=True, choices=["val", "test"])
parser.add_argument("--category", required=True, 
                    choices=["Movies_and_TV", "CDs_and_Vinyl", "Books"])
parser.add_argument("--eval", required=True, choices=["eval", "infer"])
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--his_len", type=int, default=8)

args = parser.parse_args()
category = args.category


if __name__ == "__main__":

    import vllm

    if args.eval == "infer":
        with open("results.json", "r", encoding="utf-8") as f:
            all_results = json.load(f)
        best_model = max(all_results, key=lambda x: x['meteor'])
        model_name = best_model['model']
        sampling_params = vllm.SamplingParams(
            max_tokens=2048,
            skip_special_tokens=True,
            temperature=args.temperature,
            top_p=0.95
        )
        llm = vllm.LLM(
            model_name,
            gpu_memory_utilization=0.9,
            enforce_eager=True
        )

        personal_dataset = load_from_disk(f"data/dataset_test_{category}_{args.his_len}")
        prompts = personal_dataset['inp_str']
        his_diff_embs = personal_dataset['his_diff_emb']
        his_diff_embs = [torch.tensor(his_diff_emb)
                            for his_diff_emb in his_diff_embs]
        predictions = llm.generate(
            prompts,
            his_diff_embs=his_diff_embs,
            sampling_params=sampling_params,
        )
        predictions = [pred.outputs[0].text.strip() 
                    for pred in predictions]
        if not os.path.exists("output"):
            os.makedirs("output")
        with open(f"output/predictions_{category}_{args.his_len}.txt", "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(pred + "\n---------------------------------\n")
    elif args.eval == "eval":
        personal_dataset = load_from_disk(f"data/dataset_test_{category}_{args.his_len}")
        references = personal_dataset['out_str']
        predictions_path = f"output/predictions_{category}_{args.his_len}.txt"
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = f.read()
            predictions = predictions.split('\n---------------------------------\n')
            predictions = predictions[:-1]
        predictions = [postprocess_output(prediction) for prediction in predictions]
        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load('rouge')
        meteor_metric = evaluate.load('meteor')
        result_bleu = bleu_metric.compute(predictions=predictions,
                                        references=references)
        result_rouge = rouge_metric.compute(predictions=predictions,
                                            references=references)
        result_meteor = meteor_metric.compute(predictions=predictions,
                                            references=references)
        result = {
            "rouge-1": result_rouge["rouge1"],
            "rouge-L": result_rouge["rougeL"],
            "meteor": result_meteor['meteor'],
            "bleu": result_bleu["score"],
        }
        print(result)

if dist.is_initialized():
    dist.destroy_process_group()
sys.exit(0)
