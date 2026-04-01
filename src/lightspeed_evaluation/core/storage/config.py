"""Configuration models for storage backends.

Defines Pydantic models for file and database storage configuration.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lightspeed_evaluation.core.constants import (
    DEFAULT_BASE_FILENAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STORED_CONFIGS,
    SUPPORTED_CSV_COLUMNS,
    SUPPORTED_OUTPUT_TYPES,
)


class FileBackendConfig(BaseModel):
    """Configuration for file storage backend.

    File storage outputs evaluation results to CSV, JSON, and TXT files.

    Example:
        - type: "file"
          output_dir: "./eval_output"
          base_filename: "evaluation"
          enabled_outputs: [csv, json, txt]
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["file"] = "file"
    output_dir: str = Field(
        default=DEFAULT_OUTPUT_DIR,
        description="Output directory for result files",
    )
    base_filename: str = Field(
        default=DEFAULT_BASE_FILENAME,
        description="Base filename for output files",
    )
    enabled_outputs: list[str] = Field(
        default=SUPPORTED_OUTPUT_TYPES,
        description="List of enabled output types: csv, json, txt",
    )
    csv_columns: list[str] = Field(
        default=SUPPORTED_CSV_COLUMNS,
        description="CSV columns to include in detailed results",
    )
    summary_config_sections: list[str] = Field(
        default=DEFAULT_STORED_CONFIGS,
        description="Configuration sections to include in summary reports",
    )

    @field_validator("csv_columns")
    @classmethod
    def validate_csv_columns(cls, v: list[str]) -> list[str]:
        """Validate that all CSV columns are supported."""
        for column in v:
            if column not in SUPPORTED_CSV_COLUMNS:
                raise ValueError(
                    f"Unsupported CSV column: {column}. "
                    f"Supported columns: {SUPPORTED_CSV_COLUMNS}"
                )
        return v

    @field_validator("enabled_outputs")
    @classmethod
    def validate_enabled_outputs(cls, v: list[str]) -> list[str]:
        """Validate that all enabled outputs are supported."""
        for output_type in v:
            if output_type not in SUPPORTED_OUTPUT_TYPES:
                raise ValueError(
                    f"Unsupported output type: {output_type}. "
                    f"Supported types: {SUPPORTED_OUTPUT_TYPES}"
                )
        return v


class DatabaseBackendConfig(BaseModel):
    """Configuration for database storage backend.

    Supports SQLite (local file) and remote databases (PostgreSQL, MySQL).

    Example SQLite:
        - type: "sqlite"
          database: "./eval_results.db"
          table_name: "evaluation_results"

    Example PostgreSQL:
        - type: "postgres"
          database: "evaluations"
          table_name: "evaluation_results"
          host: "localhost"
          port: 5432
          user: "admin"
          password: "secret"
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["sqlite", "postgres", "mysql"]
    database: str = Field(description="Database name or file path for SQLite")
    table_name: str = Field(
        default="evaluation_results",
        description="Table name for storing results",
    )
    host: Optional[str] = Field(default=None, description="Database host")
    port: Optional[int] = Field(default=None, description="Database port")
    user: Optional[str] = Field(default=None, description="Database user")
    password: Optional[str] = Field(default=None, description="Database password")

    @model_validator(mode="after")
    def validate_connection_fields(self) -> "DatabaseBackendConfig":
        """Validate that remote databases have required connection fields."""
        if self.type in ("postgres", "mysql"):
            missing = []
            if not self.host:
                missing.append("host")
            if not self.user:
                missing.append("user")
            if not self.password:
                missing.append("password")
            if missing:
                raise ValueError(f"{self.type} backend requires: {', '.join(missing)}")
        return self


# Discriminated union for polymorphic storage configuration
StorageBackendConfig = Annotated[
    Union[FileBackendConfig, DatabaseBackendConfig],
    Field(discriminator="type"),
]
