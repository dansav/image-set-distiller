# Image Set Distiller

Turn raw image dumps into curated collections.

**image-set-distiller** takes a large folder of images (or video frames) and refines them into a smaller, higher-quality set by removing blurry shots and near-duplicates.

## Installation

Requires Python 3.12+. Using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/dansav/image-set-distiller.git
cd image-set-distiller
uv sync
```

## Usage

```bash
# Distill a folder, keeping only the sharpest and most unique
uv run distill.py ./vacation_raw ./vacation_best

# Keep top 80% sharpest, with stricter duplicate detection
uv run distill.py ./frames ./frames_best --blur-percentile 80 --similarity 0.95

# Only remove blurry images, keep all duplicates
uv run distill.py ./captures ./captures_sharp --blur-only

# Only remove duplicates, keep blurry images
uv run distill.py ./photos ./photos_unique --dedupe-only

# Use ResNet (faster) instead of DINOv2 (more accurate)
uv run distill.py ./images ./images_best --model resnet
```

## Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--blur-percentile` | 70 | Keep top N% sharpest images |
| `--blur-threshold` | auto | Manual sharpness threshold (overrides percentile) |
| `--similarity` | 0.92 | Similarity threshold for duplicates (0.0-1.0) |
| `--model` | dinov2 | Embedding model: `dinov2` (accurate) or `resnet` (fast) |
| `--blur-only` | off | Only filter by blur, skip deduplication |
| `--dedupe-only` | off | Only deduplicate, skip blur filtering |

## How It Works

1. **Sharpness Detection**: Calculates Laplacian variance for each image. Higher variance = sharper image. Images below the threshold are marked as blurry.

2. **Duplicate Detection**: Uses DINOv2 (or ResNet) to extract visual embeddings, then clusters similar images using DBSCAN. From each cluster, only the sharpest image is kept.

3. **Output**: Copies selected images to the output folder and generates an HTML report for visual review.

## Output

After running, you'll find in your output folder:

- The distilled images
- `report.html` - Interactive visual report showing kept vs rejected images
- `metadata.json` - Full processing details in JSON format
