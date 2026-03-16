import json
import re
import random
import copy
from pathlib import Path
from typing import Dict, List, Any
import argparse

class TikZDataAugmentor:
    """
    Augment TikZ-to-Python conversion dataset through parameter perturbation.
    No LLM required - pure rule-based transformations.
    """
    
    def __init__(self, seed=42):
        random.seed(seed)
        
        # Color mapping for variations
        self.color_variations = {
            'blue': ['royalblue', 'steelblue', 'dodgerblue', 'cornflowerblue'],
            'red': ['crimson', 'firebrick', 'darkred', 'indianred'],
            'green': ['forestgreen', 'seagreen', 'darkgreen', 'mediumseagreen'],
            'yellow': ['gold', 'goldenrod', 'orange', 'darkorange'],
            'purple': ['mediumpurple', 'darkviolet', 'indigo', 'rebeccapurple'],
            'gray': ['darkgray', 'dimgray', 'slategray', 'lightslategray'],
            'black': ['black'],  # Keep black as is
            'white': ['white'],  # Keep white as is
        }
    
    def perturb_numbers(self, text: str, scale_range=(0.90, 1.10), precision=2) -> str:
        """
        Perturb all numerical values in text by a random scale factor.
        Maintains relative proportions.
        """
        def replace_number(match):
            num = float(match.group())
            # Don't perturb very small numbers (likely indices or flags)
            if abs(num) < 0.001:
                return match.group()
            scale = random.uniform(*scale_range)
            new_val = num * scale
            return f"{new_val:.{precision}f}"
        
        # Match floating point and integer numbers (including negative)
        pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        return re.sub(pattern, replace_number, text)
    
    def perturb_coordinates(self, text: str, scale_range=(0.92, 1.08)) -> str:
        """
        Specifically target coordinate pairs (x, y) for perturbation.
        More conservative to maintain visual structure.
        """
        def replace_coord_pair(match):
            x, y = float(match.group(1)), float(match.group(2))
            scale_x = random.uniform(*scale_range)
            scale_y = random.uniform(*scale_range)
            return f"({x * scale_x:.2f}, {y * scale_y:.2f})"
        
        # Match patterns like (123, 456) or (1.23, 4.56)
        pattern = r'\((-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\)'
        return re.sub(pattern, replace_coord_pair, text)
    
    def perturb_colors(self, text: str) -> str:
        """
        Replace color names with similar alternatives.
        Case-insensitive matching.
        """
        modified_text = text
        
        for base_color, variations in self.color_variations.items():
            if base_color in ['black', 'white']:
                continue  # Skip these
            
            # Find all occurrences (case insensitive)
            pattern = re.compile(re.escape(base_color), re.IGNORECASE)
            matches = list(pattern.finditer(text))
            
            if matches and variations:
                # Pick one variation for consistency within this augmentation
                new_color = random.choice(variations)
                modified_text = pattern.sub(new_color, modified_text)
        
        return modified_text
    
    def perturb_sizes(self, text: str, scale_range=(0.85, 1.15)) -> str:
        """
        Perturb size-related parameters: linewidth, markersize, fontsize, etc.
        """
        size_params = [
            r'linewidth\s*=\s*(\d+\.?\d*)',
            r'markersize\s*=\s*(\d+\.?\d*)',
            r'fontsize\s*=\s*(\d+\.?\d*)',
            r'width\s*=\s*(\d+\.?\d*)',
            r'height\s*=\s*(\d+\.?\d*)',
            r's\s*=\s*(\d+)',  # scatter size
            r'lw\s*=\s*(\d+\.?\d*)',  # linewidth shorthand
        ]
        
        modified_text = text
        for pattern in size_params:
            def scale_param(match):
                val = float(match.group(1))
                new_val = val * random.uniform(*scale_range)
                param_name = match.group(0).split('=')[0]
                return f"{param_name}={new_val:.2f}"
            
            modified_text = re.sub(pattern, scale_param, modified_text)
        
        return modified_text
    
    def perturb_opacity(self, text: str) -> str:
        """
        Adjust alpha/opacity values slightly.
        """
        def adjust_alpha(match):
            alpha = float(match.group(1))
            # Small perturbation to opacity
            new_alpha = max(0.1, min(1.0, alpha + random.uniform(-0.1, 0.1)))
            return f"alpha={new_alpha:.2f}"
        
        pattern = r'alpha\s*=\s*(\d+\.?\d*)'
        return re.sub(pattern, adjust_alpha, text)

    def _looks_like_tikz(self, text: str) -> bool:
        """Heuristic check for TikZ/LaTeX content."""
        if not text:
            return False
        t = text.lower()
        return (
            '\\begin{tikzpicture}' in t
            or '\\usetikzlibrary' in t
            or '\\documentclass' in t
            or 'pgfplots' in t
            or '\\tikz' in t
        )

    def _looks_like_python(self, text: str) -> bool:
        """Heuristic check for Python plotting/code content."""
        if not text:
            return False
        py_indicators = [
            'import ',
            'plt.',
            'np.',
            'def ',
            'print(',
            'pandas',
            'matplotlib',
            'seaborn',
            'sklearn',
            'torch',
            'tensorflow',
            'sns.',
        ]
        return any(tok in text for tok in py_indicators)
    
    def augment_sample(self, sample: Dict[str, Any], aug_id: int, 
                      perturb_coords=True, perturb_cols=True, 
                      perturb_size_params=True) -> Dict[str, Any]:
        """
        Create one augmented version of a sample.
        """
        aug_sample = copy.deepcopy(sample)

        # Handle both dataset schemas:
        # 1) TikZ schema: code=<tikz>, python_code=<python>, id=<id>
        # 2) Python-only schema: code=<python>, file_id=<id-like>
        python_field = None
        if aug_sample.get('python_code'):
            python_field = 'python_code'
        elif self._looks_like_python(str(aug_sample.get('code', ''))):
            python_field = 'code'

        if python_field:
            python_code = str(aug_sample.get(python_field, ''))
            if perturb_coords:
                python_code = self.perturb_coordinates(python_code)
            if perturb_cols:
                python_code = self.perturb_colors(python_code)
            if perturb_size_params:
                python_code = self.perturb_sizes(python_code)
                python_code = self.perturb_opacity(python_code)
            aug_sample[python_field] = python_code

        tikz_code = str(aug_sample.get('code', ''))
        if self._looks_like_tikz(tikz_code):
            if perturb_coords:
                tikz_code = self.perturb_coordinates(tikz_code)
            if perturb_cols:
                tikz_code = self.perturb_colors(tikz_code)
            aug_sample['code'] = tikz_code

        # Update ID to track augmentation (fallbacks for mixed schemas)
        original_id = aug_sample.get('id') or aug_sample.get('file_id') or f"sample_{aug_id}"
        aug_sample['id'] = f"{original_id}_aug{aug_id}"
        
        # Add metadata
        aug_sample['augmented'] = True
        aug_sample['original_id'] = original_id
        aug_sample['augmentation_type'] = 'parameter_perturbation'
        
        # Caption and other fields stay the same
        return aug_sample
    
    def augment_dataset(self, samples: List[Dict[str, Any]], 
                       variations_per_sample: int = 4,
                       include_original: bool = True) -> List[Dict[str, Any]]:
        """
        Augment entire dataset.
        
        Args:
            samples: List of original samples
            variations_per_sample: How many augmented versions to create per sample
            include_original: Whether to include original samples in output
        
        Returns:
            List of original + augmented samples
        """
        augmented_dataset = []
        
        if include_original:
            augmented_dataset.extend(samples)
        
        for i, sample in enumerate(samples):
            # Create variations
            for aug_id in range(variations_per_sample):
                # Randomly decide which perturbations to apply
                perturb_coords = random.random() > 0.2  # 80% chance
                perturb_cols = random.random() > 0.5    # 50% chance
                perturb_sizes = random.random() > 0.3   # 70% chance
                
                aug_sample = self.augment_sample(
                    sample, aug_id,
                    perturb_coords=perturb_coords,
                    perturb_cols=perturb_cols,
                    perturb_size_params=perturb_sizes
                )
                augmented_dataset.append(aug_sample)
            
            # Progress reporting
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples...")
        
        return augmented_dataset


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Save data to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    # ============================================================================
    # CONFIGURATION - SET ALL PARAMETERS HERE
    # ============================================================================
    
    # File paths
    INPUT_FILE = 'final_dataset.jsonl'
    OUTPUT_FILE = 'final_dataset_augmented.jsonl'
    
    # Augmentation settings
    VARIATIONS_PER_SAMPLE = 4        # How many augmented versions per sample
    INCLUDE_ORIGINAL = True          # Set to False to exclude original samples
    
    # Other settings
    RANDOM_SEED = 42                 # For reproducibility
    SAMPLE_LIMIT = None              # Set to a number (e.g., 100) for testing, None for all
    
    # ============================================================================
    # END CONFIGURATION
    # ============================================================================
    
    print(f"Loading dataset from {INPUT_FILE}...")
    data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(data)} samples")
    
    if SAMPLE_LIMIT:
        data = data[:SAMPLE_LIMIT]
        print(f"Limited to {len(data)} samples for testing")
    
    print(f"\nAugmenting dataset with {VARIATIONS_PER_SAMPLE} variations per sample...")
    print(f"Include original samples: {INCLUDE_ORIGINAL}")
    
    augmentor = TikZDataAugmentor(seed=RANDOM_SEED)
    
    augmented_data = augmentor.augment_dataset(
        data,
        variations_per_sample=VARIATIONS_PER_SAMPLE,
        include_original=INCLUDE_ORIGINAL
    )
    
    print(f"\nAugmentation complete!")
    print(f"Original samples: {len(data)}")
    
    if INCLUDE_ORIGINAL:
        print(f"Augmented samples: {len(augmented_data) - len(data)}")
        print(f"Total samples: {len(augmented_data)}")
        print(f"Growth factor: {len(augmented_data) / len(data):.2f}x")
    else:
        print(f"Augmented samples (only): {len(augmented_data)}")
        print(f"Note: Original samples excluded")
    
    print(f"\nSaving to {OUTPUT_FILE}...")
    save_jsonl(augmented_data, OUTPUT_FILE)
    print("Done!")


if __name__ == "__main__":
    main()
