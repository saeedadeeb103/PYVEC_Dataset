import os
import json
import random

from datasets import Dataset as HFDataset

def parse_sources(source_filter):
    valid_sources = {"arxiv", "github", "tex", "synthetic", "curated"}
    requested = set(source_filter.split("_"))
    unknown = requested - valid_sources
    if unknown:
        raise ValueError(f"Unknown source(s): {unknown}")
    return requested

def construct_prompt(entry, input_variant, all_entries_old_caption):
    if input_variant == "caption":
        caption = entry.get("caption", "").strip()
        if not caption:
            return "", all_entries_old_caption
        all_entries_old_caption += 1
        return entry.get("caption", ""), all_entries_old_caption
    elif input_variant == "text_mentions":
        return " ".join(entry.get("text_mentions", []))
    elif input_variant == "caption_text_mentions":
        caption = entry.get("caption", "")
        mentions = " ".join(entry.get("text_mentions", []))
        return f"{caption} {mentions}".strip()
    elif input_variant == "new_caption":
        return entry.get("new_caption", "")
    elif input_variant == "caption_or_new_caption":
        caption = entry.get("caption", "")
        if caption.strip():
            return caption
        return entry.get("new_caption", "")
    elif input_variant == "caption_and_new_caption":
        return None
    elif input_variant == "new_caption_comparison":
        caption = entry.get("caption", "").strip()
        if not caption:
            return ""
        return entry.get("new_caption", "")
    return ""

def get_huggingface_dataset(json_dir, input_variant, code_length, source_filter, number_samples, data_percentage, relative, compiled, debugged, seed):
    json_paths = [
        os.path.join(json_dir, fname)
        for fname in os.listdir(json_dir)
        if fname.startswith("all_") and fname.endswith(".json")
    ]
    if debugged:
        new_json_dir = json_dir.replace("/all", "/all_new")
        debugged_json_paths = [
            os.path.join(new_json_dir, fname)
            for fname in os.listdir(new_json_dir)
            if fname.startswith("all_new_") and fname.endswith(".json")
        ]
    else:
        debugged_json_paths = []
    all_paths = json_paths + debugged_json_paths
    allowed_sources = parse_sources(source_filter)
    min_len, max_len = code_length
    data = []
    all_entries = 0
    all_entries_old_caption = 0
    for path in all_paths:
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
            for entry in entries:
                all_entries += 1
                if compiled:
                    if entry.get("status") != "success":
                        continue
                if entry.get("source") not in allowed_sources:
                    continue
                cl = entry.get("code_length", 0)
                if not (min_len <= cl <= max_len):
                    continue

                if input_variant == "caption_and_new_caption":
                    caption = entry.get("caption", "").strip()
                    new_caption = entry.get("new_caption", "").strip()
                    if caption:
                        data.append({"text": caption, "label": entry["code"]})
                    if new_caption:
                        data.append({"text": new_caption, "label": entry["code"]})
                else:
                    if input_variant == "caption":
                        prompt, all_entries_old_caption = construct_prompt(entry, input_variant, all_entries_old_caption)
                    else:
                        prompt = construct_prompt(entry, input_variant, all_entries_old_caption)
                    if not prompt.strip():
                        continue
                    data.append({"text": prompt, "label": entry["code"]})
    print(f"All Samples without filtering: {all_entries}")
    if input_variant == "caption":
        print(f"All Samples without filtering (old caption): {all_entries_old_caption}")
    total = len(data)
    if relative:
        target = int(total * data_percentage)
    else:
        target = number_samples
    if target < total:
        random.seed(seed)
        data = random.sample(data, target)
    elif target > total:
        print(f"Requested {target}, but only {total} available. Using all samples.")
    return HFDataset.from_list(data)

def get_huggingface_dataset_val(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    descs, codes = [], []
    for example in data:
        desc = example["new_caption"].strip()
        code = example["code"].strip()
        descs.append(desc)
        codes.append(code)
    return HFDataset.from_dict({"text": descs, "label": codes})