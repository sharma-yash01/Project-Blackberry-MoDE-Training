# -*- coding: utf-8 -*-
"""
Vertex AI Utilities

Handles Vertex AI environment detection, output directory management,
and GCS upload functionality for training outputs.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import logging

# Try to import GCS client
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logging.warning("google-cloud-storage not available. GCS uploads will be disabled.")

logger = logging.getLogger(__name__)

# Vertex AI environment variables
AIP_MODEL_DIR = os.getenv("AIP_MODEL_DIR")
AIP_CHECKPOINT_DIR = os.getenv("AIP_CHECKPOINT_DIR")
AIP_TENSORBOARD_LOG_DIR = os.getenv("AIP_TENSORBOARD_LOG_DIR")

# Default GCS location for model outputs (fallback when AIP_MODEL_DIR is not set)
DEFAULT_MODEL_OUTPUT_BUCKET = "mode-training-init-us-central-1"
DEFAULT_MODEL_OUTPUT_PATH = "init-run/model-output"
DEFAULT_MODEL_OUTPUT_GCS = f"gs://{DEFAULT_MODEL_OUTPUT_BUCKET}/{DEFAULT_MODEL_OUTPUT_PATH}"

def is_vertex_ai_environment() -> bool:
    """
    Check if running in Vertex AI Custom Training Job.
    
    Simple check: AIP_MODEL_DIR is set (this is set by Vertex AI when base_output_dir is configured).
    This is the most reliable indicator and doesn't rely on complex environment detection.
    """
    return AIP_MODEL_DIR is not None


def should_upload_to_default_gcs() -> bool:
    """
    Determine if we should upload to default GCS location.
    
    SIMPLE, SAFE CHECK:
    - Returns True if: GCS is available AND AIP_MODEL_DIR is NOT set
    - This ensures we upload as fallback when base_output_dir is not configured
    - No complex environment detection - just checks if GCS client is available
    - Safe: If this returns True, upload will be attempted at end of training
    - Safe: Upload is wrapped in try-except, so it won't break training if it fails
    
    This is the simplest possible approach that guarantees model/metrics saving.
    """
    # Simple: If GCS is available and AIP_MODEL_DIR is not set, we should upload
    # This covers the case where base_output_dir is not set in console UI
    return GCS_AVAILABLE and (AIP_MODEL_DIR is None)


def get_vertex_ai_output_dir() -> Optional[Path]:
    """Get the Vertex AI output directory (AIP_MODEL_DIR)"""
    if AIP_MODEL_DIR:
        return Path(AIP_MODEL_DIR)
    return None


def get_vertex_ai_checkpoint_dir() -> Optional[Path]:
    """Get the Vertex AI checkpoint directory (AIP_CHECKPOINT_DIR)"""
    if AIP_CHECKPOINT_DIR:
        return Path(AIP_CHECKPOINT_DIR)
    return None


def is_gcs_path(path: str) -> bool:
    """
    Check if a path string is a GCS path (handles both gs:// and gs:/ formats).
    
    Args:
        path: Path string to check
    
    Returns:
        True if path is a GCS path, False otherwise
    """
    if not path:
        return False
    path_str = str(path).strip()
    return path_str.startswith("gs://") or path_str.startswith("gs:/")


def resolve_output_path(requested_path: str, fallback_type: str = "checkpoint") -> Path:
    """
    Resolve output path for Vertex AI or local execution.
    
    Args:
        requested_path: Path requested by user (can be local or GCS gs:// path)
        fallback_type: Type of output ('checkpoint', 'log', 'model')
    
    Returns:
        Resolved local path that should be used for saving files
        
    Strategy:
        1. If running on Vertex AI, use AIP_MODEL_DIR (Vertex AI syncs this to GCS automatically)
        2. If local execution with GCS path, extract local equivalent path or use fallback
        3. If local execution with local path, use as-is
        4. Always validate that returned path is a valid local path
    """
    # Normalize requested_path - handle empty strings
    if not requested_path:
        requested_path = "./outputs" if fallback_type == "checkpoint" else "./logs"
    
    requested_path = str(requested_path).strip()
    
    # Check if requested_path is a GCS path (handle both gs:// and gs:/)
    is_gcs = is_gcs_path(requested_path)
    
    # Check if we're on Vertex AI
    if is_vertex_ai_environment():
        # Use AIP_MODEL_DIR if available (Vertex AI automatically syncs this to base_output_dir)
        # AIP_MODEL_DIR is a LOCAL path that Vertex AI mounts from GCS
        # CRITICAL: When AIP_MODEL_DIR is set, we MUST use it as a local path, not a GCS path!
        if AIP_MODEL_DIR:
            # Validate AIP_MODEL_DIR is actually a local path (not a GCS path)
            aip_model_dir_str = str(AIP_MODEL_DIR).strip()
            if is_gcs_path(aip_model_dir_str):
                logger.error(f"ERROR: AIP_MODEL_DIR is set to a GCS path: {AIP_MODEL_DIR}")
                logger.error("AIP_MODEL_DIR must be a local path. Using fallback local path.")
                # Fall back to a local temp directory
                fallback_dir = Path("/tmp") / "vertex_ai_outputs" / fallback_type
                fallback_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Using fallback local path: {fallback_dir}")
                return fallback_dir
            
            vertex_dir = Path(AIP_MODEL_DIR)
            
            # CRITICAL FIX: When AIP_MODEL_DIR is set, ignore GCS paths in requested_path
            # Vertex AI will sync AIP_MODEL_DIR to the base_output_dir automatically
            if is_gcs:
                # User passed a GCS path, but AIP_MODEL_DIR is set - ignore the GCS path
                logger.warning(
                    f"GCS path '{requested_path}' provided but AIP_MODEL_DIR is set. "
                    f"Using AIP_MODEL_DIR ({AIP_MODEL_DIR}) as local mount point. "
                    f"Vertex AI will sync to your base_output_dir automatically."
                )
                # For checkpoints, use AIP_MODEL_DIR directly (no subdirectory)
                # For other types, optionally create a subdirectory
                if fallback_type != "checkpoint":
                    # Extract last directory name from GCS path as subdirectory
                    # Remove gs:// or gs:/ prefix and split
                    normalized = requested_path.replace("gs://", "").replace("gs:/", "")
                    parts = [p for p in normalized.split("/") if p]
                    if len(parts) > 0:
                        subdir_name = parts[-1]
                        vertex_dir = vertex_dir / subdir_name
                        logger.info(f"  Creating subdirectory '{subdir_name}' within AIP_MODEL_DIR for {fallback_type}")
            elif fallback_type == "checkpoint":
                # For checkpoint type with non-GCS path, use AIP_MODEL_DIR directly
                # This handles: "./checkpoints", "checkpoints", ".", "", etc.
                # No need to create subdirectories - use AIP_MODEL_DIR root
                pass
            else:
                # For non-checkpoint types with non-GCS paths, optionally extract directory name
                req_path = Path(requested_path)
                if req_path.name and req_path.name not in [".", ""]:
                    vertex_dir = vertex_dir / req_path.name
            
            # Create directory and validate it's a local path
            try:
                vertex_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {vertex_dir}: {e}")
                # Fall back to temp directory
                fallback_dir = Path("/tmp") / "vertex_ai_outputs" / fallback_type
                fallback_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Using fallback local path: {fallback_dir}")
                return fallback_dir
            
            # CRITICAL VALIDATION: Ensure resolved path is a valid local path
            resolved_str = str(vertex_dir)
            if is_gcs_path(resolved_str):
                logger.error(f"ERROR: Resolved path is still a GCS path: {resolved_str}")
                logger.error("This should not happen! Using fallback local path.")
                fallback_dir = Path("/tmp") / "vertex_ai_outputs" / fallback_type
                fallback_dir.mkdir(parents=True, exist_ok=True)
                return fallback_dir
            
            logger.info(f"Vertex AI: Using LOCAL path {vertex_dir} for {fallback_type} outputs")
            logger.info(f"  AIP_MODEL_DIR: {AIP_MODEL_DIR}")
            logger.info(f"  Resolved path: {vertex_dir}")
            logger.info(f"  Vertex AI will automatically sync this to your configured GCS bucket")
            return vertex_dir
        else:
            # AIP_MODEL_DIR is not set, but we're in Vertex AI environment
            # This is unusual - use a fallback local path
            logger.warning("AIP_MODEL_DIR is not set in Vertex AI environment. Using fallback local path.")
            fallback_dir = Path("/tmp") / "vertex_ai_outputs" / fallback_type
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir
    
    # Local execution - handle GCS paths by converting to local equivalents
    if is_gcs:
        # Extract a local path equivalent from GCS path
        # Normalize: remove gs:// or gs:/ prefix
        normalized = requested_path.replace("gs://", "").replace("gs:/", "")
        parts = [p for p in normalized.split("/") if p]
        
        if len(parts) > 1:
            # Skip bucket name, use rest as local path
            # e.g., "gs://bucket/checkpoints/mode" -> "./checkpoints/mode"
            local_path = Path("./") / "/".join(parts[1:])
        elif len(parts) == 1:
            # Only bucket name - use default local path
            local_path = Path("./outputs") / fallback_type
        else:
            # Empty path - use default
            local_path = Path("./outputs") / fallback_type
        
        logger.info(f"Local execution with GCS path: Using {local_path} (will sync to GCS after training)")
    else:
        # Local path - use as-is, but ensure it's not accidentally a GCS path
        # This can happen if Path() is created from a string that looks like GCS
        req_path_str = str(requested_path)
        if is_gcs_path(req_path_str):
            logger.error(f"ERROR: Path appears to be GCS path: {req_path_str}")
            logger.error("Using fallback local path instead.")
            local_path = Path("./outputs") / fallback_type
        else:
            local_path = Path(requested_path)
    
    # Validate final path is not a GCS path
    final_path_str = str(local_path)
    if is_gcs_path(final_path_str):
        logger.error(f"ERROR: Final resolved path is a GCS path: {final_path_str}")
        logger.error("This should not happen! Using fallback local path.")
        local_path = Path("./outputs") / fallback_type
    
    # Create directory
    try:
        local_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory {local_path}: {e}")
        # Fall back to temp directory
        fallback_dir = Path("/tmp") / "training_outputs" / fallback_type
        fallback_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Using fallback local path: {fallback_dir}")
        return fallback_dir
    
    # Final validation: ensure path is absolute or relative local path (not GCS)
    final_str = str(local_path.resolve() if local_path.exists() else local_path)
    if is_gcs_path(final_str):
        logger.error(f"ERROR: Resolved absolute path is a GCS path: {final_str}")
        fallback_dir = Path("/tmp") / "training_outputs" / fallback_type
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir
    
    return local_path


def upload_file_to_gcs(
    local_file_path: str,
    gcs_bucket_name: str,
    gcs_blob_path: str,
    project_id: Optional[str] = None
) -> bool:
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        local_file_path: Path to local file to upload
        gcs_bucket_name: GCS bucket name (without gs:// prefix)
        gcs_blob_path: Path within bucket (e.g., 'checkpoints/mode/metrics_history.csv')
        project_id: GCP project ID (optional, will use default if not provided)
    
    Returns:
        True if upload successful, False otherwise
    """
    if not GCS_AVAILABLE:
        logger.warning("GCS client not available. Cannot upload file.")
        return False
    
    try:
        # Initialize GCS client
        if project_id:
            storage_client = storage.Client(project=project_id)
        else:
            storage_client = storage.Client()
        
        # Get bucket
        bucket = storage_client.bucket(gcs_bucket_name)
        
        # Create blob
        blob = bucket.blob(gcs_blob_path)
        
        # Upload file
        blob.upload_from_filename(local_file_path)
        
        logger.info(f"✓ Uploaded {local_file_path} to gs://{gcs_bucket_name}/{gcs_blob_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to upload {local_file_path} to GCS: {e}")
        return False


def upload_directory_to_gcs(
    local_dir_path: str,
    gcs_bucket_name: str,
    gcs_prefix: str,
    project_id: Optional[str] = None,
    exclude_patterns: Optional[list] = None
) -> bool:
    """
    Upload a directory to Google Cloud Storage recursively.
    
    Args:
        local_dir_path: Path to local directory to upload
        gcs_bucket_name: GCS bucket name (without gs:// prefix)
        gcs_prefix: Prefix for GCS paths (e.g., 'checkpoints/mode')
        project_id: GCP project ID (optional)
        exclude_patterns: List of file patterns to exclude (e.g., ['*.pyc', '__pycache__'])
    
    Returns:
        True if upload successful, False otherwise
    """
    if not GCS_AVAILABLE:
        logger.warning("GCS client not available. Cannot upload directory.")
        return False
    
    if exclude_patterns is None:
        exclude_patterns = ['*.pyc', '__pycache__', '*.py', '.git']
    
    try:
        # Initialize GCS client
        if project_id:
            storage_client = storage.Client(project=project_id)
        else:
            storage_client = storage.Client()
        
        # Get bucket
        bucket = storage_client.bucket(gcs_bucket_name)
        
        local_path = Path(local_dir_path)
        uploaded_count = 0
        
        # Walk through directory
        for root, dirs, files in os.walk(local_dir_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(
                Path(d).match(pattern) for pattern in exclude_patterns
            )]
            
            for file in files:
                # Check if file should be excluded
                if any(Path(file).match(pattern) for pattern in exclude_patterns):
                    continue
                
                local_file = Path(root) / file
                # Get relative path from local_dir_path
                relative_path = local_file.relative_to(local_path)
                # Construct GCS blob path
                gcs_blob_path = f"{gcs_prefix}/{relative_path}".replace("\\", "/")
                
                # Upload file
                blob = bucket.blob(gcs_blob_path)
                blob.upload_from_filename(str(local_file))
                uploaded_count += 1
        
        logger.info(
            f"✓ Uploaded {uploaded_count} files from {local_dir_path} "
            f"to gs://{gcs_bucket_name}/{gcs_prefix}"
        )
        return True
    
    except Exception as e:
        logger.error(f"Failed to upload directory {local_dir_path} to GCS: {e}")
        return False


def sync_to_gcs_if_needed(
    local_path: str,
    requested_gcs_path: Optional[str] = None,
    bucket_name: Optional[str] = None,
    project_id: Optional[str] = None,
    is_directory: bool = False,
    force_upload: bool = False
) -> bool:
    """
    Sync a file or directory to GCS if needed.
    
    This function checks if:
    1. We're on Vertex AI with AIP_MODEL_DIR set (automatic sync, skip upload)
    2. A GCS path was requested or force_upload is True
    3. GCS is available
    
    If all conditions are met, uploads to GCS.
    
    Args:
        local_path: Local file or directory path
        requested_gcs_path: Original GCS path requested (gs://bucket/path)
        bucket_name: GCS bucket name (extracted from requested_gcs_path if not provided)
        project_id: GCP project ID
        is_directory: Whether local_path is a directory
        force_upload: If True, upload even if on Vertex AI (useful when AIP_MODEL_DIR not set)
    
    Returns:
        True if sync successful or not needed, False if sync failed
    """
    # If we're on Vertex AI with AIP_MODEL_DIR set, it's automatically synced
    # Only skip if AIP_MODEL_DIR is set (automatic sync) and not forcing upload
    if is_vertex_ai_environment() and AIP_MODEL_DIR is not None and not force_upload:
        logger.debug("Running on Vertex AI with AIP_MODEL_DIR set. Vertex AI will automatically sync to GCS.")
        return True
    
    # If no GCS path was requested and not forcing upload, nothing to sync
    if not requested_gcs_path or not requested_gcs_path.startswith("gs://"):
        if not force_upload:
            return True
        # If forcing upload but no path provided, we can't proceed
        if force_upload and not bucket_name:
            logger.warning("force_upload=True but no GCS path or bucket_name provided. Cannot upload.")
            return False
    
    # Extract bucket and path from GCS URI
    if bucket_name:
        # Bucket name provided explicitly
        if requested_gcs_path and requested_gcs_path.startswith("gs://"):
            # Extract path after bucket from full GCS URI
            if requested_gcs_path.startswith(f"gs://{bucket_name}/"):
                gcs_path = requested_gcs_path.replace(f"gs://{bucket_name}/", "")
            else:
                # Different bucket in path - extract just the path part
                parts = requested_gcs_path.replace("gs://", "").split("/", 1)
                gcs_path = parts[1] if len(parts) > 1 else ""
        else:
            # No requested_gcs_path, but bucket_name provided - use local path name or provided path
            # This happens when we call upload_file_to_gcs directly with bucket and blob path
            if requested_gcs_path and not requested_gcs_path.startswith("gs://"):
                # Use requested_gcs_path as the blob path directly
                gcs_path = requested_gcs_path
            else:
                # Fallback to local file name
                gcs_path = Path(local_path).name
    else:
        # Extract bucket from GCS URI
        if not requested_gcs_path or not requested_gcs_path.startswith("gs://"):
            logger.warning(f"Cannot extract bucket name: requested_gcs_path={requested_gcs_path}, bucket_name={bucket_name}")
            return False
        parts = requested_gcs_path.replace("gs://", "").split("/", 1)
        if len(parts) < 1:
            logger.warning(f"Invalid GCS path: {requested_gcs_path}")
            return False
        bucket_name = parts[0]
        gcs_path = parts[1] if len(parts) > 1 else ""
    
    # Upload to GCS
    if is_directory:
        return upload_directory_to_gcs(local_path, bucket_name, gcs_path, project_id)
    else:
        # For files, use gcs_path directly as the blob path
        # gcs_path already contains the full path within the bucket
        return upload_file_to_gcs(local_path, bucket_name, gcs_path, project_id)

