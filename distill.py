#!/usr/bin/env python3
"""
Image Set Distiller

Turn raw image dumps into curated collections by removing blurry shots
and near-duplicates, keeping only the sharpest, most unique frames.

Usage:
    uv run distill.py <input_folder> <output_folder> [options]

Example:
    uv run distill.py ./vacation_raw ./vacation_best --blur-percentile 70 --similarity 0.92
"""

import argparse
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm


@dataclass
class ImageInfo:
    """Metadata about a processed image."""
    path: Path
    sharpness: float
    embedding: Optional[np.ndarray] = None
    cluster_id: int = -1
    kept: bool = False
    kept_reason: str = ""


def calculate_sharpness(image_path: Path) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    Higher values = sharper image.
    """
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)

        # Proper Laplacian kernel convolution
        # Kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        h, w = img_array.shape
        laplacian = np.zeros_like(img_array)

        # Apply Laplacian (avoiding edges)
        laplacian[1:-1, 1:-1] = (
            img_array[0:-2, 1:-1] +  # top
            img_array[2:, 1:-1] +    # bottom
            img_array[1:-1, 0:-2] +  # left
            img_array[1:-1, 2:] -    # right
            4 * img_array[1:-1, 1:-1]
        )

        return float(np.var(laplacian))
    except Exception as e:
        print(f"Warning: Could not process {image_path.name}: {e}")
        return 0.0


class EmbeddingModel:
    """Wrapper for different embedding models."""

    def __init__(self, model_name: str = "dinov2", device: str = None):
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                  else "cuda" if torch.cuda.is_available()
                                  else "cpu")
        print(f"Using device: {self.device}")
        self._load_model()

    def _load_model(self):
        """Load the selected model."""
        if self.model_name == "dinov2":
            # DINOv2 - excellent for fine-grained visual similarity
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.eval()
            self.model.to(self.device)
            self.input_size = 518  # DINOv2 preferred size (14*37)
            self.normalize_mean = [0.485, 0.456, 0.406]
            self.normalize_std = [0.229, 0.224, 0.225]

        elif self.model_name == "resnet":
            # ResNet50 - fast and reliable
            from torchvision import models
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Remove classification layer, keep feature extractor
            self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.model.eval()
            self.model.to(self.device)
            self.input_size = 224
            self.normalize_mean = [0.485, 0.456, 0.406]
            self.normalize_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def get_embedding(self, image_path: Path) -> np.ndarray:
        """Extract embedding from an image."""
        try:
            img = Image.open(image_path).convert('RGB')

            # Resize and center crop
            img = img.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)

            # Convert to tensor and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = (img_array - self.normalize_mean) / self.normalize_std
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                embedding = self.model(img_tensor)

            # Flatten and normalize
            embedding = embedding.squeeze().cpu().numpy()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            return embedding

        except Exception as e:
            print(f"Warning: Could not get embedding for {image_path.name}: {e}")
            return np.zeros(384 if self.model_name == "dinov2" else 2048)


def find_images(directory: Path) -> list[Path]:
    """Find all image files in a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    images = []
    for ext in extensions:
        images.extend(directory.glob(f'*{ext}'))
        images.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(images)


def filter_by_sharpness(
    images: list[ImageInfo],
    threshold: float = None,
    percentile: float = 70.0
) -> tuple[list[ImageInfo], float]:
    """
    Filter images by sharpness, marking blurry ones.

    Returns:
        Tuple of (all images with updated kept status, threshold used)
    """
    sharpness_values = [img.sharpness for img in images]

    if threshold is None:
        threshold = float(np.percentile(sharpness_values, 100 - percentile))

    for img in images:
        if img.sharpness >= threshold:
            img.kept = True
            img.kept_reason = "sharp"
        else:
            img.kept = False
            img.kept_reason = "blurry"

    return images, threshold


def deduplicate_by_similarity(
    images: list[ImageInfo],
    similarity_threshold: float = 0.92,
    model: EmbeddingModel = None
) -> list[ImageInfo]:
    """
    Remove near-duplicate images, keeping the sharpest from each cluster.
    Only processes images that are marked as kept (sharp).
    """
    sharp_images = [img for img in images if img.kept]

    if not sharp_images:
        return images

    print(f"\nExtracting embeddings from {len(sharp_images)} sharp images...")
    for img in tqdm(sharp_images):
        img.embedding = model.get_embedding(img.path)

    # Build embedding matrix
    embeddings = np.array([img.embedding for img in sharp_images])

    # DBSCAN with cosine distance
    # eps = 1 - similarity_threshold (cosine distance)
    eps = 1 - similarity_threshold
    clustering = DBSCAN(eps=eps, min_samples=1, metric='cosine')
    labels = clustering.fit_predict(embeddings)

    # Assign cluster IDs
    for img, label in zip(sharp_images, labels):
        img.cluster_id = int(label)

    # For each cluster, keep only the sharpest image
    clusters = {}
    for img in sharp_images:
        if img.cluster_id not in clusters:
            clusters[img.cluster_id] = []
        clusters[img.cluster_id].append(img)

    kept_count = 0
    duplicate_count = 0

    for cluster_id, cluster_images in clusters.items():
        # Sort by sharpness, keep the sharpest
        cluster_images.sort(key=lambda x: x.sharpness, reverse=True)

        for i, img in enumerate(cluster_images):
            if i == 0:
                img.kept = True
                img.kept_reason = "unique" if len(cluster_images) == 1 else "sharpest_in_cluster"
                kept_count += 1
            else:
                img.kept = False
                img.kept_reason = f"duplicate_of_{cluster_images[0].path.name}"
                duplicate_count += 1

    print(f"Found {len(clusters)} unique scenes, removed {duplicate_count} duplicates")

    return images


def create_html_report(
    images: list[ImageInfo],
    output_dir: Path,
    sharpness_threshold: float,
    similarity_threshold: float
):
    """Generate an HTML report for visual review."""

    kept = [img for img in images if img.kept]
    rejected = [img for img in images if not img.kept]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Image Set Distiller Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        h1 {{ color: #fff; border-bottom: 2px solid #4a4a6a; padding-bottom: 10px; }}
        .stats {{
            background: #16213e;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 30px;
        }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #00d9ff; }}
        .stat-label {{ color: #888; font-size: 0.9em; }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            background: #16213e;
            border: none;
            color: #888;
            cursor: pointer;
            border-radius: 5px;
            font-size: 1em;
        }}
        .tab.active {{ background: #0f3460; color: #00d9ff; }}
        .tab:hover {{ background: #0f3460; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }}
        .card {{
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s;
        }}
        .card:hover {{ transform: scale(1.02); }}
        .card img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            cursor: pointer;
        }}
        .card-info {{
            padding: 10px;
            font-size: 0.85em;
        }}
        .card-info .filename {{
            font-weight: bold;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .card-info .metrics {{ color: #888; margin-top: 5px; }}
        .card-info .reason {{
            margin-top: 5px;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            display: inline-block;
        }}
        .reason.sharp {{ background: #1b4332; color: #95d5b2; }}
        .reason.unique {{ background: #1b4332; color: #95d5b2; }}
        .reason.sharpest_in_cluster {{ background: #0f3460; color: #00d9ff; }}
        .reason.blurry {{ background: #5c2a2a; color: #f8a5a5; }}
        .reason.duplicate {{ background: #5c4a2a; color: #f8d9a5; }}
        .section {{ display: none; }}
        .section.active {{ display: block; }}
        .modal {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        .modal.active {{ display: flex; }}
        .modal img {{ max-width: 90vw; max-height: 90vh; }}
        .modal-close {{
            position: fixed;
            top: 20px;
            right: 30px;
            color: #fff;
            font-size: 2em;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <h1>Image Set Distiller Report</h1>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">{len(images)}</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(kept)}</div>
            <div class="stat-label">Kept</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(rejected)}</div>
            <div class="stat-label">Rejected</div>
        </div>
        <div class="stat">
            <div class="stat-value">{100 * len(kept) / len(images):.1f}%</div>
            <div class="stat-label">Retention Rate</div>
        </div>
        <div class="stat">
            <div class="stat-value">{sharpness_threshold:.1f}</div>
            <div class="stat-label">Sharpness Threshold</div>
        </div>
        <div class="stat">
            <div class="stat-value">{similarity_threshold:.2f}</div>
            <div class="stat-label">Similarity Threshold</div>
        </div>
    </div>

    <div class="tabs">
        <button class="tab active" onclick="showSection('kept')">✓ Kept ({len(kept)})</button>
        <button class="tab" onclick="showSection('rejected')">✗ Rejected ({len(rejected)})</button>
        <button class="tab" onclick="showSection('all')">All ({len(images)})</button>
    </div>

    <div id="kept" class="section active">
        <div class="grid">
"""

    # Sort kept images by sharpness
    for img in sorted(kept, key=lambda x: x.sharpness, reverse=True):
        reason_class = "unique" if "unique" in img.kept_reason else "sharpest_in_cluster" if "sharpest" in img.kept_reason else "sharp"
        html += f"""
            <div class="card">
                <img src="../{output_dir.name}/{img.path.name}" onclick="openModal(this.src)" loading="lazy">
                <div class="card-info">
                    <div class="filename">{img.path.name}</div>
                    <div class="metrics">Sharpness: {img.sharpness:.1f}</div>
                    <span class="reason {reason_class}">{img.kept_reason}</span>
                </div>
            </div>
"""

    html += """
        </div>
    </div>

    <div id="rejected" class="section">
        <div class="grid">
"""

    for img in sorted(rejected, key=lambda x: x.sharpness, reverse=True):
        reason_class = "blurry" if "blurry" in img.kept_reason else "duplicate"
        # Link to original folder for rejected images
        html += f"""
            <div class="card">
                <img src="{img.path}" onclick="openModal(this.src)" loading="lazy">
                <div class="card-info">
                    <div class="filename">{img.path.name}</div>
                    <div class="metrics">Sharpness: {img.sharpness:.1f}</div>
                    <span class="reason {reason_class}">{img.kept_reason}</span>
                </div>
            </div>
"""

    html += """
        </div>
    </div>

    <div id="all" class="section">
        <div class="grid">
"""

    for img in sorted(images, key=lambda x: x.path.name):
        if img.kept:
            reason_class = "unique" if "unique" in img.kept_reason else "sharpest_in_cluster" if "sharpest" in img.kept_reason else "sharp"
            img_src = f"../{output_dir.name}/{img.path.name}"
        else:
            reason_class = "blurry" if "blurry" in img.kept_reason else "duplicate"
            img_src = str(img.path)
        html += f"""
            <div class="card">
                <img src="{img_src}" onclick="openModal(this.src)" loading="lazy">
                <div class="card-info">
                    <div class="filename">{img.path.name}</div>
                    <div class="metrics">Sharpness: {img.sharpness:.1f}</div>
                    <span class="reason {reason_class}">{img.kept_reason}</span>
                </div>
            </div>
"""

    html += """
        </div>
    </div>

    <div class="modal" id="modal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img id="modal-img" src="">
    </div>

    <script>
        function showSection(id) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            event.target.classList.add('active');
        }

        function openModal(src) {
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').classList.add('active');
            event.stopPropagation();
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') closeModal();
        });
    </script>
</body>
</html>
"""

    report_path = output_dir / "report.html"
    with open(report_path, 'w') as f:
        f.write(html)

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Distill image collections by removing blurry and duplicate images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./vacation_raw ./vacation_best
  %(prog)s ./frames ./frames_best --blur-percentile 80 --similarity 0.95
  %(prog)s ./captures ./captures_distilled --model resnet --blur-only
        """
    )
    parser.add_argument("input", type=Path, help="Input folder containing images")
    parser.add_argument("output", type=Path, help="Output folder for reduced images")
    parser.add_argument("--blur-percentile", type=float, default=70.0,
                        help="Keep top N%% sharpest images (default: 70)")
    parser.add_argument("--blur-threshold", type=float, default=None,
                        help="Manual sharpness threshold (overrides percentile)")
    parser.add_argument("--similarity", type=float, default=0.92,
                        help="Similarity threshold for duplicates (0.0-1.0, default: 0.92)")
    parser.add_argument("--model", choices=["dinov2", "resnet"], default="dinov2",
                        help="Embedding model to use (default: dinov2)")
    parser.add_argument("--blur-only", action="store_true",
                        help="Only filter by blur, skip deduplication")
    parser.add_argument("--dedupe-only", action="store_true",
                        help="Only deduplicate, skip blur filtering")

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input folder does not exist: {args.input}")
        return 1

    # Find images
    image_paths = find_images(args.input)
    if not image_paths:
        print(f"Error: No images found in {args.input}")
        return 1

    print(f"Found {len(image_paths)} images in {args.input}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Step 1: Calculate sharpness for all images
    print("\nCalculating sharpness scores...")
    images = []
    for path in tqdm(image_paths):
        sharpness = calculate_sharpness(path)
        images.append(ImageInfo(path=path, sharpness=sharpness, kept=True))

    sharpness_threshold = 0.0

    # Step 2: Filter by sharpness (unless --dedupe-only)
    if not args.dedupe_only:
        print(f"\nFiltering by sharpness (keeping top {args.blur_percentile}%)...")
        images, sharpness_threshold = filter_by_sharpness(
            images,
            threshold=args.blur_threshold,
            percentile=args.blur_percentile
        )
        sharp_count = sum(1 for img in images if img.kept)
        print(f"Kept {sharp_count}/{len(images)} sharp images (threshold: {sharpness_threshold:.1f})")

    # Step 3: Deduplicate (unless --blur-only)
    if not args.blur_only:
        print(f"\nLoading {args.model} model for similarity detection...")
        model = EmbeddingModel(model_name=args.model)
        images = deduplicate_by_similarity(
            images,
            similarity_threshold=args.similarity,
            model=model
        )

    # Step 4: Copy kept images to output
    kept_images = [img for img in images if img.kept]
    print(f"\nCopying {len(kept_images)} images to {args.output}...")
    for img in tqdm(kept_images):
        shutil.copy2(img.path, args.output / img.path.name)

    # Step 5: Generate report
    print("\nGenerating HTML report...")
    report_path = create_html_report(
        images,
        args.output,
        sharpness_threshold,
        args.similarity
    )

    # Step 6: Save JSON metadata
    metadata = {
        "input_folder": str(args.input.absolute()),
        "output_folder": str(args.output.absolute()),
        "total_images": len(images),
        "kept_images": len(kept_images),
        "rejected_images": len(images) - len(kept_images),
        "sharpness_threshold": sharpness_threshold,
        "similarity_threshold": args.similarity,
        "model": args.model,
        "images": [
            {
                "filename": img.path.name,
                "sharpness": img.sharpness,
                "cluster_id": img.cluster_id,
                "kept": img.kept,
                "reason": img.kept_reason
            }
            for img in sorted(images, key=lambda x: x.path.name)
        ]
    }

    with open(args.output / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"""
{'='*50}
Summary:
  Input:     {len(images)} images
  Output:    {len(kept_images)} images ({100*len(kept_images)/len(images):.1f}%)
  Removed:   {len(images) - len(kept_images)} images

  Report:    {report_path}
  Metadata:  {args.output / 'metadata.json'}
{'='*50}
""")

    return 0


if __name__ == "__main__":
    exit(main())
