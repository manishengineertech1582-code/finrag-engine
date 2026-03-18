"""
Environment Configuration Utility

This module provides safe access to environment variables,
specifically for sensitive credentials like API keys.

IMPORTANT:
- Never log or print full API keys in production.
- Always mask sensitive values.
"""

import os
import logging
from typing import Optional

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def _mask_key(key: str, visible_chars: int = 4) -> str:
    """
    Mask a sensitive key for safe logging.

    Example:
        sk-1234567890abcdef -> sk-12**********cdef

    Args:
        key: Full API key
        visible_chars: Number of characters to show at start/end

    Returns:
        Masked key string
    """
    if not key or len(key) <= visible_chars * 2:
        return "****"

    return f"{key[:visible_chars]}{'*' * (len(key) - 2 * visible_chars)}{key[-visible_chars:]}"


def get_openai_api_key(required: bool = True) -> Optional[str]:
    """
    Retrieve OpenAI API key from environment variables.

    Args:
        required: If True, raises an error when key is missing.

    Returns:
        API key string if found, otherwise None.

    Raises:
        EnvironmentError: If key is required but not set.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        message = "OPENAI_API_KEY not found in environment variables."

        if required:
            logger.error(message)
            raise EnvironmentError(message)

        logger.warning(message)
        return None

    # Safe logging (masked key only)
    logger.info("OpenAI API key loaded: %s", _mask_key(api_key))

    return api_key


# -------------------------------------------------------------------
# Script Entry Point (for debugging only)
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        key = get_openai_api_key(required=False)

        if key:
            print("API Key loaded successfully (masked).")
        else:
            print("API Key not found.")

    except Exception as e:
        logger.exception("Error while retrieving API key.")
        print(f"Error: {str(e)}")