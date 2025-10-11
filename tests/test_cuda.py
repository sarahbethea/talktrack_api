# Run with:
#   python -m tests.test_cuda
# or:
#   pytest -q tests/test_cuda.py

"""
Quick environment sanity checks:
- Torch CUDA availability (if torch is installed)
- Hugging Face authentication (if HF_TOKEN is set)
"""

import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()  # loads HF_TOKEN if present

def test_cuda_info() -> None:
    """Log basic CUDA info if torch is available; otherwise skip gracefully."""
    try:
        import torch  # local import so this test doesn't force torch to be installed
    except Exception as e:
        logger.warning("torch not installed or failed to import (%s) — skipping CUDA check.", e)
        return

    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("CUDA device count: %s", torch.cuda.device_count())

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            logger.info("GPU name: %s", name)
        except Exception as e:
            logger.warning("Could not query GPU name: %s", e)


def test_huggingface_auth() -> None:
    """Log current HF user if HF_TOKEN is present; otherwise skip gracefully."""
    token = os.getenv("HF_TOKEN")  # keep naming consistent with the rest of the repo
    if not token:
        logger.warning("HF_TOKEN not set — skipping Hugging Face whoami.")
        return

    try:
        from huggingface_hub import whoami
        info = whoami(token)
        # Avoid logging sensitive fields; show minimal identity
        user = info.get("name") or info.get("email") or info.get("orgs", ["<unknown>"])[0]
        logger.info("Hugging Face auth OK (user/org: %s)", user)
    except Exception as e:
        logger.error("Hugging Face whoami failed: %s", e)


if __name__ == "__main__":
    test_cuda_info()
    test_huggingface_auth()
