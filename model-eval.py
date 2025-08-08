import os
import sys
import torch
import argparse
import evaluate
import warnings
import numpy as np
import torch.distributed as dist

from datasets import load_from_disk
from vllm import LLM, SamplingParams
from bert_score import score as bert_score
from transformers import set_seed, logging

from utils.utils import postprocess_output

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--dataset", required=True, choices=["val", "test"])
parser.add_argument("--category", required=True, 
                    choices=["Movies_and_TV", "CDs_and_Vinyl", "Books"])
parser.add_argument("--eval", required=True, choices=["eval", "infer"])
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--his_len", type=int, default=8)
parser.add_argument("--gpu", default="0")

args = parser.parse_args()
category = args.category

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if __name__ == "__main__":

    if args.eval == "infer":
        # with open("results.json", "r", encoding="utf-8") as f:
        #     all_results = json.load(f)
        # best_model = max(all_results, key=lambda x: x['meteor'])
        # model_name = best_model['model']
        model_name = "SnowCharmQ/DEP-model"
        sampling_params = SamplingParams(
            max_tokens=2048,
            skip_special_tokens=True,
            temperature=args.temperature,
            top_p=0.95
        )
        llm = LLM(
            model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            enforce_eager=True
        )

        personal_dataset = load_from_disk(f"data/dataset_test_{category}")
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
        with open(f"output/predictions_{category}.txt", "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(pred + "\n---------------------------------\n")
    elif args.eval == "eval":
        personal_dataset = load_from_disk(f"data/dataset_test_{category}")
        references = personal_dataset['out_str']
        references = list(references)
        predictions_path = f"output/predictions_{category}.txt"
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = f.read()
            predictions = predictions.split('\n---------------------------------\n')
            predictions = predictions[:-1]
        predictions = [postprocess_output(prediction) for prediction in predictions]
        bleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load('rouge')
        meteor_metric = evaluate.load('meteor')
        result_bleu = bleu_metric.compute(
            predictions=predictions,
            references=references
        )
        result_rouge = rouge_metric.compute(
            predictions=predictions,
            references=references
        )
        result_meteor = meteor_metric.compute(
            predictions=predictions,
            references=references
        )
        P, R, F1 = bert_score(
            predictions, 
            references, 
            model_type="allenai/led-base-16384",
            lang="en", 
            verbose=False,
        )
        result = {
            "rouge-1": result_rouge["rouge1"].item() if isinstance(result_rouge["rouge1"], np.float64) else result_rouge["rouge1"],
            "rouge-L": result_rouge["rougeL"].item() if isinstance(result_rouge["rougeL"], np.float64) else result_rouge["rougeL"],
            "meteor": result_meteor['meteor'].item() if isinstance(result_meteor['meteor'], np.float64) else result_meteor['meteor'],
            "bleu": result_bleu["score"].item() if isinstance(result_bleu["score"], np.float64) else result_bleu["score"],
            "bertscore": F1.mean().item(),
        }
        print(result)

if dist.is_initialized():
    dist.destroy_process_group()
sys.exit(0)
