"""
deploy_to_huggingface.py

Automated deployment tool to package and upload the OphthalmoAI ML backend
and model weights to Hugging Face Spaces using the official huggingface_hub SDK.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("Error: 'huggingface_hub' library not found. Install with: pip install huggingface_hub")
    sys.exit(1)


ROOT_DIR = Path(__file__).resolve().parent.parent


def prepare_staging_dir(staging_path: Path) -> None:
    """Prepares the exact folder structure required by Hugging Face Spaces."""
    print("-> Assembling deployment package...")

    # Copy Hugging Face metadata and Dockerfile
    hf_source_dir = ROOT_DIR / "deploy" / "huggingface"
    shutil.copy2(hf_source_dir / "Dockerfile", staging_path / "Dockerfile")
    shutil.copy2(hf_source_dir / "README.md", staging_path / "README.md")
    shutil.copy2(hf_source_dir / ".dockerignore", staging_path / ".dockerignore")

    # Copy alembic migrations
    shutil.copy2(ROOT_DIR / "alembic.ini", staging_path / "alembic.ini")
    shutil.copytree(
        ROOT_DIR / "alembic",
        staging_path / "alembic",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    # Copy backend code
    shutil.copytree(
        ROOT_DIR / "backend",
        staging_path / "backend",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.db", "tests*"),
    )

    # Copy trained model weights
    models_target = staging_path / "models"
    models_source = ROOT_DIR / "models"
    if not models_source.exists():
        raise FileNotFoundError(f"Models directory not found at {models_source}")

    shutil.copytree(
        models_source,
        models_target,
        ignore=shutil.ignore_patterns("*.tmp", "*checkpoint*"),
    )

    print(f"-> Package assembled at {staging_path}")
    model_count = len(list(models_target.glob("*.pth")))
    print(f"   Included {model_count} trained neural network weights (.pth)")


def deploy(repo_id: str, token: str, dry_run: bool = False) -> None:
    """Deploys the packaged backend to Hugging Face Spaces."""
    print(f"\n==========================================")
    print(f"Deploying OphthalmoAI to Hugging Face Spaces")
    print(f"Target Space: {repo_id}")
    print(f"==========================================\n")

    with tempfile.TemporaryDirectory(prefix="ophthalmoai_hf_") as temp_dir:
        staging_path = Path(temp_dir)
        prepare_staging_dir(staging_path)

        if dry_run:
            print("\n[DRY RUN] Package successfully prepared. No files uploaded.")
            return

        api = HfApi(token=token)

        print("\n-> Ensuring Space exists on Hugging Face...")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="docker",
                exist_ok=True,
                private=False,
            )
            print(f"-> Space '{repo_id}' ready.")
        except Exception as exc:
            print(f"Notice during space verification: {exc}")

        print("\n-> Uploading files and model weights (this may take a few minutes)...")
        api.upload_folder(
            folder_path=str(staging_path),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy OphthalmoAI backend inference engine",
        )

        user, space_name = repo_id.split("/", 1) if "/" in repo_id else ("user", repo_id)
        # Hugging Face Spaces subdomains convert underscores/slashes to hyphens
        hf_subdomain = f"{user}-{space_name}".replace("_", "-").lower()
        public_api_url = f"https://{hf_subdomain}.hf.space"

        print("\n" + "=" * 50)
        print("SUCCESS! Files uploaded to Hugging Face Spaces.")
        print("=" * 50)
        print(f"\nYour Space URL: https://huggingface.co/spaces/{repo_id}")
        print(f"Your Public API URL (once running): {public_api_url}")
        print("\nNext steps in your Space Settings (Variables & Secrets):")
        print("1. Set Secret: JWT_SECRET_KEY = <random-32-byte-hex>")
        print("2. Set Secret: GEMINI_API_KEY = <your-google-gemini-key>")
        print("3. Set Variable: ENVIRONMENT = production")
        print("4. Set Variable: CORS_ORIGINS = * (or your Vercel URL once deployed)")
        print("5. Set Variable: PERSIST_SCANS = false")
        print("\nSet this API URL in your Vercel frontend project:")
        print(f"VITE_API_URL={public_api_url}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Deploy OphthalmoAI Backend to Hugging Face Spaces")
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Hugging Face Space ID in the format 'username/space-name' (e.g. akash/ophthalmoai-backend)",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face User Access Token (with write permission). Defaults to $HF_TOKEN env var.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Package files locally to verify without uploading to Hugging Face.",
    )
    args = parser.parse_args()

    repo_id = args.repo_id
    if not repo_id:
        repo_id = input("Enter your Hugging Face Space ID (username/space-name): ").strip()
        if not repo_id or "/" not in repo_id:
            print("Error: Space ID must be in format 'username/space-name'")
            sys.exit(1)

    token = args.token or os.getenv("HF_TOKEN")
    if not token and not args.dry_run:
        token = getpass.getpass("Enter your Hugging Face Access Token (write permission): ").strip()
        if not token:
            print("Error: Hugging Face token is required for upload.")
            sys.exit(1)

    deploy(repo_id=repo_id, token=token, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
