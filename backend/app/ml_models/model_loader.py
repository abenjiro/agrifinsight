"""
Model Loader for Cloud-Hosted Models
Downloads and caches AI models from cloud storage (S3, Hugging Face, etc.)
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Downloads and caches ML models from cloud storage
    Supports S3, Google Cloud Storage, Hugging Face, direct URLs, etc.
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize model loader

        Args:
            cache_dir: Directory to cache downloaded models (default: /tmp/models)
        """
        if cache_dir is None:
            cache_dir = os.getenv('MODEL_CACHE_DIR', '/tmp/models')

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model cache directory: {self.cache_dir}")

    def get_model_path(
        self,
        model_url: str,
        model_name: str = None,
        force_download: bool = False
    ) -> str:
        """
        Get model file path, downloading if necessary

        Args:
            model_url: URL to download model from (S3, GCS, HTTP, etc.)
            model_name: Optional custom name for cached file
            force_download: Force re-download even if cached

        Returns:
            Path to the model file
        """
        # Generate cache filename
        if model_name is None:
            # Use hash of URL as filename
            url_hash = hashlib.md5(model_url.encode()).hexdigest()
            # Get extension from URL
            parsed = urlparse(model_url)
            extension = Path(parsed.path).suffix or '.pth'
            model_name = f"model_{url_hash}{extension}"

        cache_path = self.cache_dir / model_name

        # Check if already cached
        if cache_path.exists() and not force_download:
            logger.info(f"Using cached model: {cache_path}")
            return str(cache_path)

        # Download model
        logger.info(f"Downloading model from: {model_url}")
        try:
            self._download_file(model_url, cache_path)
            logger.info(f"Model downloaded successfully: {cache_path}")
            return str(cache_path)

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # If download fails but cached version exists, use it
            if cache_path.exists():
                logger.warning("Using existing cached model after download failure")
                return str(cache_path)
            raise

    def _download_file(self, url: str, destination: Path, chunk_size: int = 8192):
        """
        Download file from URL with progress logging

        Args:
            url: URL to download from
            destination: Path to save file
            chunk_size: Size of chunks to download
        """
        # Handle S3 presigned URLs and regular HTTP URLs
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress
        downloaded = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress every 10MB
                    if total_size > 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)")

        logger.info(f"Download complete: {downloaded / (1024*1024):.1f} MB")

    def download_from_s3(
        self,
        bucket: str,
        key: str,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        region: str = 'us-east-1'
    ) -> str:
        """
        Download model directly from S3 using boto3

        Args:
            bucket: S3 bucket name
            key: S3 object key (path to model file)
            aws_access_key_id: AWS access key (optional, uses env vars if not provided)
            aws_secret_access_key: AWS secret key (optional)
            region: AWS region

        Returns:
            Path to downloaded model file
        """
        try:
            import boto3

            # Initialize S3 client
            if aws_access_key_id and aws_secret_access_key:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region
                )
            else:
                # Use environment variables or IAM role
                s3_client = boto3.client('s3', region_name=region)

            # Generate cache filename
            model_name = Path(key).name
            cache_path = self.cache_dir / model_name

            # Check if already cached
            if cache_path.exists():
                logger.info(f"Using cached model: {cache_path}")
                return str(cache_path)

            # Download from S3
            logger.info(f"Downloading from S3: s3://{bucket}/{key}")
            s3_client.download_file(bucket, key, str(cache_path))
            logger.info(f"Model downloaded successfully: {cache_path}")

            return str(cache_path)

        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            raise
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            raise

    def download_from_huggingface(
        self,
        repo_id: str,
        filename: str,
        token: str = None
    ) -> str:
        """
        Download model from Hugging Face Hub

        Args:
            repo_id: Hugging Face repository ID (e.g., "username/model-name")
            filename: Name of the model file in the repo
            token: Hugging Face API token (optional, for private repos)

        Returns:
            Path to downloaded model file
        """
        try:
            from huggingface_hub import hf_hub_download

            logger.info(f"Downloading from Hugging Face: {repo_id}/{filename}")

            cache_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                cache_dir=str(self.cache_dir)
            )

            logger.info(f"Model downloaded successfully: {cache_path}")
            return cache_path

        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            raise
        except Exception as e:
            logger.error(f"Failed to download from Hugging Face: {e}")
            raise


# Singleton instance
_model_loader = None


def get_model_loader() -> ModelLoader:
    """Get or create singleton model loader instance"""
    global _model_loader

    if _model_loader is None:
        _model_loader = ModelLoader()
        logger.info("Model loader singleton created")

    return _model_loader
