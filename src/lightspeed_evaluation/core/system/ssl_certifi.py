"""Combines certifi's default certificates with custom and system certificates."""

import atexit
import os
from typing import Any
from pathlib import Path
import tempfile
import certifi


def get_ssl_cert_files_paths_from_system_yaml(config_data: dict[str, Any]) -> list[str]:
    """From `system config yaml file` get all ssl_cert_file values where ssl_verify is True."""
    custom_cert_paths = []

    # Check if this dict has both ssl_verify and ssl_cert_file
    if config_data.get("ssl_verify"):
        ssl_cert = config_data.get("ssl_cert_file")
        if ssl_cert is not None:
            custom_cert_paths.append(ssl_cert)

    # Recurse into nested dicts
    for v in config_data.values():
        if isinstance(v, dict):
            custom_cert_paths.extend(get_ssl_cert_files_paths_from_system_yaml(v))

    return custom_cert_paths


def get_system_ssl_cert_file() -> list[str]:
    """Get system SSL cert file path from environment variable."""
    ssl_cert_file = os.environ.get("SSL_CERT_FILE")
    if ssl_cert_file:
        return [ssl_cert_file]
    return []


def _get_unique_ssl_cert_paths(cert_paths: list[str]) -> list[str]:
    """Return unique SSL certificate paths from the provided list."""
    return list(set(cert_paths))


def create_ssl_certifi_bundle(config_data: dict[str, Any]) -> str:
    """Combines system certs (from certifi) with environment andcustom certs."""
    # Get custom cert paths from config
    custom_cert_paths = get_ssl_cert_files_paths_from_system_yaml(config_data)

    # Get SSL_CERT_FILE environment variable
    system_cert_file = get_system_ssl_cert_file()

    # Combine all cert paths
    cert_paths = custom_cert_paths + system_cert_file
    # Keep only unique paths
    cert_paths = _get_unique_ssl_cert_paths(cert_paths)

    # If no custom certs, just return certifi's bundle
    if not cert_paths:
        return certifi.where()

    # Read certifi's default bundle
    certifi_bundle = Path(certifi.where()).read_text(encoding="utf-8")

    # Read custom certificates
    combined_certs = "\n".join(
        Path(cert_path).read_text(encoding="utf-8") for cert_path in cert_paths
    )

    # Create temporary combined bundle
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".pem"
    ) as combined_bundle:
        combined_bundle.write(certifi_bundle)
        combined_bundle.write("\n")
        combined_bundle.write(combined_certs)
        bundle_path = combined_bundle.name

    # Register cleanup on program exit
    atexit.register(lambda: Path(bundle_path).unlink(missing_ok=True))

    return bundle_path
