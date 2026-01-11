"""
Utility for downloading ProcessTransformer models from HuggingFace Hub.

Can be used standalone to pre-download models before simulation.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(
    repo_id: str,
    revision: str = "main",
    cache_dir: str = None,
    force: bool = False,
):
    """
    Download ProcessTransformer model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        revision: Git revision/tag
        cache_dir: Local cache directory
        force: If True, re-download even if cached
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "bpic17-simulation" / "process_transformer" / revision
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    files = ["model.keras", "vocab.json", "config.json", "training_metadata.json"]

    for filename in files:
        local_path = cache_dir / filename

        if local_path.exists() and not force:
            logger.info(f"Already cached: {filename}")
            continue

        logger.info(f"Downloading: {filename}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
            logger.info(f"Downloaded to: {local_path}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            if filename == "training_metadata.json":
                logger.info("(training_metadata.json is optional)")
            else:
                raise

    logger.info(f"\nModel cached at: {cache_dir}")
    return cache_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ProcessTransformer model")
    parser.add_argument(
        "--repo-id",
        default="lgk03/bpic17-process-transformer_v1",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision or tag",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local cache directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download",
    )

    args = parser.parse_args()
    download_model(args.repo_id, args.revision, args.cache_dir, args.force)

