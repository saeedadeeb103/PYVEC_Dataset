import os
import json
import tarfile

from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datasets import Dataset, concatenate_datasets

def get_huggingface_dataset_val(image_dir, json_path, code_length):
    with open(json_path, "r") as f:
        full_data = json.load(f)
    examples = []
    for entry in tqdm(full_data, desc=f"Matching {json_path}"):
        tikz_code = entry.get("code")
        figure_id = entry.get("figure_id")
        if tikz_code is None or figure_id is None:
            continue
        if code_length is not None and not (code_length[0] <= len(tikz_code) <= code_length[1]):
            continue
        image_path = os.path.join(image_dir, f"{figure_id}.png")
        if not os.path.exists(image_path):
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        examples.append({
            "image": image,
            "text": tikz_code
        })
    return Dataset.from_list(examples)

def process_metadata_tar(json_path, tar_path, code_length):
    examples = []
    try:
        with open(json_path, "r") as f:
            entries = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decode failed for {json_path}: {e}")
        return examples
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            images = {}
            for m in tar.getmembers():
                if not m.isfile() or not m.name.endswith(".png"):
                    continue
                file_id = os.path.basename(m.name).replace(".png", "")
                try:
                    img_bytes = tar.extractfile(m).read()
                    image = Image.open(BytesIO(img_bytes)).convert("RGB")
                    images[file_id] = image
                except Exception as e:
                    print(f"Failed to load image {m.name}: {e}")
                    continue
            for entry in entries:
                tikz_code = entry.get("code")
                file_id = entry.get("file_id")
                if (
                    tikz_code is None
                    or file_id is None
                    or not (code_length[0] <= len(tikz_code) <= code_length[1])
                    or file_id not in images
                ):
                    continue
                examples.append({
                    "text": tikz_code,
                    "image": images[file_id]
                })
    except Exception as e:
        print(f"Failed to process tarball {tar_path}: {e}")
    return examples

def save_huggingface_dataset_chunks_streamed(json_dir, code_length, tmp_dir):
    output_dir = os.path.join(tmp_dir, "chunks")
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(tmp_dir, "unified_dataset", "images")
    json_files = sorted([
        fname for fname in os.listdir(json_dir)
        if fname.startswith("metadata_") and fname.endswith(".json")
    ])
    chunk_paths = []
    for idx, fname in tqdm(list(enumerate(json_files)), desc="Streaming tar+json to disk"):
        archive_id = fname[len("metadata_"):-len(".json")]
        json_path = os.path.join(json_dir, fname)
        tar_path = os.path.join(image_dir, f"images_{archive_id}.tar.gz")
        if not os.path.exists(tar_path):
            print(f"Tarball not found: {tar_path}")
            continue
        try:
            examples = process_metadata_tar(json_path, tar_path, code_length)
            if not examples:
                continue
            dataset = Dataset.from_list(examples)
            chunk_path = os.path.join(output_dir, f"chunk_{idx:04d}")
            dataset.save_to_disk(chunk_path)
            chunk_paths.append(chunk_path)
        except Exception as e:
            print(f"Chunk {idx} failed: {e}")
    return chunk_paths

def load_saved_chunks(chunk_paths):
    datasets = [Dataset.load_from_disk(path) for path in chunk_paths]
    return concatenate_datasets(datasets)