#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

try:
    import openslide
except ImportError as error:
    raise ImportError("openslide-python is required to process .svs files") from error

try:
    from dgl.data.utils import save_graphs
except ImportError as error:
    raise ImportError("dgl is required to save graph artifacts") from error

sys.path.append(os.path.join(os.getcwd(), 'histocartography'))
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder
from histocartography.visualization import OverlayGraphVisualization


THUMBNAIL_SIZE = (4096, 4096)
MIN_NUCLEI_THRESHOLD = 5


def is_critical_exception(error: Exception) -> bool:
    if isinstance(error, MemoryError):
        return True
    if isinstance(error, RuntimeError):
        return "out of memory" in str(error).lower()
    return False


def get_wsi_files(input_dir: Path, file_list: str | None) -> list[Path]:
    if file_list:
        files = []
        with open(file_list, "r", encoding="utf-8") as file:
            for line in file:
                raw_path = line.strip()
                if not raw_path:
                    continue
                path = Path(raw_path)
                if not path.exists():
                    print(f"Skipping missing file in file list: {path}")
                    continue
                if path.suffix.lower() != ".svs":
                    print(f"Skipping non-SVS entry in file list: {path}")
                    continue
                files.append(path)
        return files
    return sorted(input_dir.glob("*.svs"))


def extract_wsi_artifacts(
    wsi_path: Path,
    output_dir: Path,
    nuclei_detector: NucleiExtractor,
    feature_extractor: DeepFeatureExtractor,
    knn_graph_builder: KNNGraphBuilder,
    visualizer: OverlayGraphVisualization,
) -> dict:
    thumbnail_dir = output_dir / "thumbnails"
    graph_dir = output_dir / "cell_graphs"
    graph_viz_dir = output_dir / "cell_graphs_viz"
    embedding_dir = output_dir / "node_embeddings"
    nuclei_summary_dir = output_dir / "nuclei_summaries"

    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_viz_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    nuclei_summary_dir.mkdir(parents=True, exist_ok=True)

    slide_id = wsi_path.stem
    thumbnail_path = thumbnail_dir / f"{slide_id}.png"
    graph_path = graph_dir / f"{slide_id}.bin"
    graph_viz_path = graph_viz_dir / f"{slide_id}.png"
    embedding_path = embedding_dir / f"{slide_id}.npy"
    nuclei_summary_path = nuclei_summary_dir / f"{slide_id}.npz"

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        thumbnail = slide.get_thumbnail(THUMBNAIL_SIZE).convert("RGB")
    finally:
        slide.close()

    thumbnail.save(thumbnail_path)
    image_np = np.array(thumbnail)

    nuclei_map, nuclei_centers = nuclei_detector.process(image_np)
    np.savez_compressed(
        nuclei_summary_path,
        nuclei_count=int(nuclei_centers.shape[0]),
        nuclei_map=nuclei_map,
    )

    result = {
        "slide_id": slide_id,
        "wsi_path": str(wsi_path),
        "thumbnail_path": str(thumbnail_path),
        "graph_path": None,
        "graph_viz_path": None,
        "embedding_path": None,
        "nuclei_summary_path": str(nuclei_summary_path),
        "nuclei_count": int(nuclei_centers.shape[0]),
        "embedding_shape": None,
        "graph_nodes": 0,
        "status": "skipped",
        "skip_reason": None,
        "error": None,
    }

    if nuclei_centers.shape[0] <= MIN_NUCLEI_THRESHOLD:
        result["skip_reason"] = f"insufficient nuclei ({nuclei_centers.shape[0]})"
        return result

    features = feature_extractor.process(image_np, nuclei_map)
    node_embeddings = features.detach().cpu().numpy()

    graph = knn_graph_builder.process(nuclei_map, features)
    save_graphs(str(graph_path), [graph])

    canvas = visualizer.process(image_np, graph, instance_map=nuclei_map)
    canvas.save(graph_viz_path)

    np.save(embedding_path, node_embeddings)

    result["graph_path"] = str(graph_path)
    result["graph_viz_path"] = str(graph_viz_path)
    result["embedding_path"] = str(embedding_path)
    result["embedding_shape"] = list(node_embeddings.shape)
    result["graph_nodes"] = int(graph.number_of_nodes())
    result["status"] = "processed"
    return result


def write_aggregate_bundle(records: list[dict], output_dir: Path) -> Path:
    bundle = {
        "records": records,
        "slide_to_embeddings": {},
        "slide_to_graph_path": {},
        "slide_to_viz_path": {},
        "slide_to_thumbnail_path": {},
        "slide_to_nuclei_summary_path": {},
        "processed_slides": [],
        "skipped_slides": [],
        "failed_slides": [],
    }

    for record in records:
        slide_id = record["slide_id"]
        bundle["slide_to_thumbnail_path"][slide_id] = record["thumbnail_path"]
        bundle["slide_to_nuclei_summary_path"][slide_id] = record["nuclei_summary_path"]

        if record["status"] == "processed":
            bundle["processed_slides"].append(slide_id)
            bundle["slide_to_graph_path"][slide_id] = record["graph_path"]
            bundle["slide_to_viz_path"][slide_id] = record["graph_viz_path"]
            bundle["slide_to_embeddings"][slide_id] = np.load(record["embedding_path"])
        elif record["status"] == "skipped":
            bundle["skipped_slides"].append(
                {
                    "slide_id": slide_id,
                    "reason": record["skip_reason"],
                    "nuclei_count": record["nuclei_count"],
                }
            )
        else:
            bundle["failed_slides"].append(
                {
                    "slide_id": slide_id,
                    "error": record["error"],
                }
            )

    bundle_path = output_dir / "wsi_graph_bundle.pkl"
    with open(bundle_path, "wb") as file:
        pickle.dump(bundle, file)
    return bundle_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Process WSI files into thumbnails, graphs, and node embeddings")
    parser.add_argument("--input_dir", required=True, help="Directory containing .svs files")
    parser.add_argument("--output_dir", required=True, help="Directory for output artifacts")
    parser.add_argument("--temp_dir", default="/tmp", help="Temporary directory")
    parser.add_argument("--file_list", help="Optional file listing absolute or relative .svs paths")
    parser.add_argument("--thumbnail_size", type=int, default=4096, help="Square thumbnail size in pixels")
    parser.add_argument("--k", type=int, default=5, help="K for KNN graph builder")
    parser.add_argument("--thresh", type=float, default=50, help="Distance threshold for graph edges")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch size for DeepFeatureExtractor")
    parser.add_argument("--resize_size", type=int, default=224, help="Resize size for DeepFeatureExtractor")
    args = parser.parse_args()

    global THUMBNAIL_SIZE
    THUMBNAIL_SIZE = (args.thumbnail_size, args.thumbnail_size)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp_dir)

    wsi_files = get_wsi_files(input_dir, args.file_list)
    print(f"Found {len(wsi_files)} WSI files")
    if not wsi_files:
        return

    nuclei_detector = NucleiExtractor()
    feature_extractor = DeepFeatureExtractor(
        architecture='resnet34',
        patch_size=args.patch_size,
        resize_size=args.resize_size,
    )
    knn_graph_builder = KNNGraphBuilder(k=args.k, thresh=args.thresh, add_loc_feats=True)
    visualizer = OverlayGraphVisualization()

    records = []
    for wsi_path in wsi_files:
        try:
            record = extract_wsi_artifacts(
                wsi_path=wsi_path,
                output_dir=output_dir,
                nuclei_detector=nuclei_detector,
                feature_extractor=feature_extractor,
                knn_graph_builder=knn_graph_builder,
                visualizer=visualizer,
            )
            records.append(record)
            if record["status"] == "processed":
                print(
                    f"Processed {wsi_path.name}: nuclei={record['nuclei_count']}, "
                    f"nodes={record['graph_nodes']}, emb_shape={tuple(record['embedding_shape'])}"
                )
            else:
                print(f"Skipped {wsi_path.name}: {record['skip_reason']}")
        except Exception as error:
            print(f"ERROR processing {wsi_path} ({type(error).__name__}): {error}")
            traceback.print_exc()
            records.append(
                {
                    "slide_id": wsi_path.stem,
                    "wsi_path": str(wsi_path),
                    "thumbnail_path": None,
                    "graph_path": None,
                    "graph_viz_path": None,
                    "embedding_path": None,
                    "nuclei_summary_path": None,
                    "nuclei_count": 0,
                    "embedding_shape": None,
                    "graph_nodes": 0,
                    "status": "failed",
                    "skip_reason": None,
                    "error": str(error),
                }
            )
            if is_critical_exception(error):
                print("Critical error detected. Failing fast.")
                raise

    bundle_path = write_aggregate_bundle(records, output_dir)
    processed = sum(1 for rec in records if rec["status"] == "processed")
    skipped = sum(1 for rec in records if rec["status"] == "skipped")
    failed = sum(1 for rec in records if rec["status"] == "failed")
    print(
        f"Done. processed={processed}, skipped={skipped}, failed={failed}. "
        f"Aggregate bundle: {bundle_path}"
    )


if __name__ == "__main__":
    main()
