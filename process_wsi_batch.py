#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import openslide
except ImportError as error:
    raise ImportError("openslide-python is required to process .svs files") from error

from histocartography.preprocessing import NucleiExtractor


def get_wsi_files(input_dir: Path, file_list: str | None) -> list[Path]:
    if file_list:
        with open(file_list, "r", encoding="utf-8") as file:
            return [Path(line.strip()) for line in file if line.strip()]
    return sorted(input_dir.glob("*.svs"))


def extract_top_nuclei_patches(wsi_path: Path, output_dir: Path, patch_count: int = 6) -> None:
    slide = openslide.OpenSlide(str(wsi_path))
    thumbnail = slide.get_thumbnail((4096, 4096)).convert("RGB")
    image = np.array(thumbnail)

    nuclei_map, nuclei_centers = NucleiExtractor().process(image)
    if nuclei_centers.shape[0] <= 5:
        print(f"Skipping {wsi_path.name}: insufficient nuclei ({nuclei_centers.shape[0]})")
        return

    width, height = thumbnail.size
    width_range = np.linspace(0, width, 4, dtype=int)
    height_range = np.linspace(0, height, 4, dtype=int)

    overlap_percent = 20
    width_overlap = int((overlap_percent / 100) * width)
    height_overlap = int((overlap_percent / 100) * height)

    patch_boxes = []
    patch_counts = []

    for i in range(len(width_range) - 1):
        for j in range(len(height_range) - 1):
            start_width = width_range[i] - width_overlap if i != 0 else width_range[i]
            start_height = height_range[j] - height_overlap if j != 0 else height_range[j]

            left = start_width
            upper = start_height
            right = width_range[i + 1]
            lower = height_range[j + 1]

            count = 0
            for center in nuclei_centers:
                if left <= center[0] <= right and upper <= center[1] <= lower:
                    count += 1

            patch_boxes.append((left, upper, right, lower))
            patch_counts.append(count)

    sorted_indices = np.flip(np.argsort(patch_counts))
    slide_output_dir = output_dir / wsi_path.stem
    slide_output_dir.mkdir(parents=True, exist_ok=True)

    for patch_index in range(min(patch_count, len(patch_boxes))):
        box = patch_boxes[sorted_indices[patch_index]]
        patch = thumbnail.crop(box)
        patch.save(slide_output_dir / f"{patch_index + 1}.png")

    np.savez_compressed(slide_output_dir / "nuclei_summary.npz", nuclei_count=nuclei_centers.shape[0], nuclei_map=nuclei_map)
    print(f"Completed {wsi_path.name} -> {slide_output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process WSI files for nuclei extraction")
    parser.add_argument("--input_dir", required=True, help="Directory containing .svs files")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument("--temp_dir", default="/tmp", help="Temporary directory (reserved for future use)")
    parser.add_argument("--file_list", help="Optional file with list of WSIs to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wsi_files = get_wsi_files(input_dir, args.file_list)
    print(f"Found {len(wsi_files)} WSI files")

    for wsi_path in wsi_files:
        try:
            extract_top_nuclei_patches(wsi_path, output_dir)
        except Exception as error:
            print(f"ERROR processing {wsi_path}: {error}")


if __name__ == "__main__":
    main()
