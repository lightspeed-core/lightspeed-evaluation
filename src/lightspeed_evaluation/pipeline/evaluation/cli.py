"""CLI client abstraction for Kubernetes cluster operations."""

from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Optional


class CLIClient(ABC):
    """Interface for CLI operations against a Kubernetes cluster."""

    def __init__(self, timeout: int) -> None:
        """Initialize with a command timeout in seconds."""
        self._timeout = timeout

    @abstractmethod
    def run(
        self,
        args: list[str],
        stdin: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a CLI command and return the completed process."""

    @abstractmethod
    def get_resource(
        self,
        resource_plural: str,
        name: str,
    ) -> tuple[dict[str, Any], Optional[str]]:
        """Fetch a single resource by name.

        Returns:
            Tuple of (resource_dict, error_message). On success the dict is
            the full JSON object; on failure the dict is empty.
        """

    @abstractmethod
    def apply(
        self,
        manifest: dict[str, Any],
    ) -> subprocess.CompletedProcess[str]:
        """Apply a manifest via stdin."""

    @abstractmethod
    def delete(self, resource_plural: str, name: str) -> None:
        """Delete a resource by name (idempotent)."""


class KubeCLI(CLIClient):
    """Concrete CLIClient backed by oc or kubectl."""

    def __init__(self, cli_path: str, namespace: str, timeout: int) -> None:
        """Initialize with a resolved binary path and target namespace."""
        super().__init__(timeout)
        self._cli = cli_path
        self._namespace = namespace

    def run(
        self,
        args: list[str],
        stdin: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a CLI command and return the completed process."""
        try:
            return subprocess.run(
                [self._cli, *args],
                input=stdin,
                text=True,
                capture_output=True,
                env=os.environ.copy(),
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            cmd_str = " ".join([self._cli, *args])
            return subprocess.CompletedProcess(
                args=[self._cli, *args],
                returncode=1,
                stdout="",
                stderr=f"Command timed out after {self._timeout}s: {cmd_str}",
            )

    def get_resource(
        self,
        resource_plural: str,
        name: str,
    ) -> tuple[dict[str, Any], Optional[str]]:
        """Fetch a single resource by name.

        Returns:
            Tuple of (resource_dict, error_message). On success the dict is
            the full JSON object; on failure the dict is empty.
        """
        result = self.run(
            ["get", resource_plural, name, "-n", self._namespace, "-o", "json"]
        )
        if result.returncode != 0:
            return (
                {},
                f"Failed to get {resource_plural}/{name}: " f"{result.stderr.strip()}",
            )
        try:
            return json.loads(result.stdout), None
        except json.JSONDecodeError as exc:
            return (
                {},
                f"Failed to parse JSON for {resource_plural}/{name}: {exc}",
            )

    def apply(
        self,
        manifest: dict[str, Any],
    ) -> subprocess.CompletedProcess[str]:
        """Apply a manifest via stdin."""
        return self.run(["apply", "-f", "-"], stdin=json.dumps(manifest))

    def delete(self, resource_plural: str, name: str) -> None:
        """Delete a resource by name (idempotent)."""
        self.run(
            [
                "delete",
                resource_plural,
                name,
                "-n",
                self._namespace,
                "--ignore-not-found",
            ]
        )
