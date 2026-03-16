"""Rejection Sampling Generation for DeTikZify (Expert Iteration)

Loads an SFT DeTikZify model, generates N candidate Python scripts per training image,
executes them, scores them with DINOv2, and saves the *best* execution per image
as a new HuggingFace dataset for the final SFT stage.

Usage:
    python generate_rejection_samples.py \
        --model_path <path_to_sft_lora_or_base> \
        --base_model_path <path_to_detikzify_base> \
        --render_dir <path_to_rendered_images> \
        --output_dataset_path <path_to_save_new_hf_dataset> \
        --num_candidates 8 \
        --temperature 1.0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

import torch
from datasets import load_from_disk, Dataset, DatasetDict
from PIL import Image, ImageOps
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_detikzify_vis import apply_lora
from eval_sft_vis import execute_python_to_image, extract_code, IMAGE_SIZE
from rewards import DINOv2Scorer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to SFT model or LoRA adapter")
    parser.add_argument("--base_model_path", required=True, help="Path to DeTikZify base if model_path is LoRA")
    parser.add_argument("--data_dir", default="data/python_ft", help="Path to original HF dataset")
    parser.add_argument("--render_dir", required=True, help="Path to GT rendered images")
    parser.add_argument("--output_dataset_path", required=True, help="Path to save filtered dataset")
    parser.add_argument("--num_candidates", type=int, default=8, help="Generations per image")
    parser.add_argument("--temperature", type=float, default=1.0, help="High temp for diversity")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base DeTikZify from {args.base_model_path}...")
    
    # Custom loading to avoid detikzify package issues
    from detikzify.model import register
    register()
    
    processor = AutoProcessor.from_pretrained(args.base_model_path, use_fast=False)
    
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if args.model_path != args.base_model_path and os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        print(f"Applying SFT LoRA from {args.model_path}...")
        model = apply_lora(model, args.model_path)
    model.eval()

    print("Loading DINOv2 scorer...")
    scorer = DINOv2Scorer(device=device)

    print(f"Loading original dataset from {args.data_dir}...")
    original_dataset = load_from_disk(args.data_dir)
    train_ds = original_dataset["train"]
    
    print(f"Starting Rejection Sampling for {len(train_ds)} training images...")
    print(f"Generating {args.num_candidates} candidates per image at temperature {args.temperature}")
    
    jsonl_path = args.output_dataset_path + ".jsonl"
    processed_ids = set()
    if os.path.exists(jsonl_path):
        print(f"Resuming from existing JSONL at {jsonl_path}...")
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["sample_id"])
                except:
                    continue
        print(f"Found {len(processed_ids)} already processed samples.")

    success_count = len(processed_ids)
    
    # We'll use a jsonl file for incremental saving to survive timeouts
    f_jsonl = open(jsonl_path, "a")

    for i in tqdm(range(len(train_ds)), desc="Processing Images"):
        sample = train_ds[i]
        sample_id = sample["sample_id"]
        
        if sample_id in processed_ids:
            continue

        caption = sample["text"]

        gt_path = os.path.join(args.render_dir, f"{sample_id}.png")
        if not os.path.exists(gt_path):
            continue
            
        try:
            gt_img = Image.open(gt_path).convert("RGB")
            gt_img_padded = ImageOps.pad(gt_img, IMAGE_SIZE, color="white")
        except Exception as e:
            print(f"Error loading GT image {sample_id}: {e}")
            continue

        # Generate candidates
        inputs = processor(
            text="", # DeTikZify prompt is just the image token inserted by processor
            images=gt_img,
            return_tensors="pt"
        ).to(model.device)

        best_score = -1.0
        best_code = None

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                num_return_sequences=args.num_candidates
            )
        
        # Process all candidates generated in the batch
        for out_seq in out:
            gen_text = processor.decode(out_seq, skip_special_tokens=True)
            candidate_code = extract_code(gen_text)

            # Execute & Score
            pred_img, error = execute_python_to_image(candidate_code)
            
            if pred_img and not error:
                score = scorer.score([pred_img], [gt_img_padded])[0]
                if score > best_score:
                    best_score = score
                    best_code = candidate_code

        # If we found at least one successful execution, keep the best one
        if best_code is not None:
            # Incremental save
            entry = {
                "sample_id": sample_id,
                "text": caption,
                "python_code": best_code,
                "dino_score": float(best_score)
            }
            f_jsonl.write(json.dumps(entry) + "\n")
            f_jsonl.flush()
            success_count += 1
            
        if (i+1) % 50 == 0:
            print(f"Progress: {i+1}/{len(train_ds)} | Rejection Sampling Success Rate: {success_count}/{(i+1)} ({(success_count/(i+1))*100:.1f}%)")

    f_jsonl.close()
    print(f"\nFinal Rejection Sampling Success: {success_count}/{len(train_ds)} ({(success_count/len(train_ds))*100:.1f}%)")
    
    print(f"Converting JSONL to final HF dataset at {args.output_dataset_path}...")
    
    # Re-reading for final conversion
    final_data = {
        "sample_id": [],
        "text": [],
        "python_code": [],
        "rendered_image": [], # We will re-load from disk for the HF dataset
        "dino_score": []
    }
    
    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Building Final Dataset"):
            data = json.loads(line)
            final_data["sample_id"].append(data["sample_id"])
            final_data["text"].append(data["text"])
            final_data["python_code"].append(data["python_code"])
            final_data["dino_score"].append(data["dino_score"])
            
            # Load the image for the HF dataset
            img_path = os.path.join(args.render_dir, f"{data['sample_id']}.png")
            final_data["rendered_image"].append(Image.open(img_path).convert("RGB"))

    new_train_dataset = Dataset.from_dict(final_data)
    
    # We keep the original validation set for fair evaluation later
    new_dataset_dict = DatasetDict({
        "train": new_train_dataset,
        "validation": original_dataset["validation"]
    })
    
    new_dataset_dict.save_to_disk(args.output_dataset_path)
    print("Done! You can now run SFT on this new dataset.")

if __name__ == "__main__":
    main()
