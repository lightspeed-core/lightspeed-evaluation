"""Tests for data persistence utilities."""

from pathlib import Path

import yaml

from lightspeed_evaluation.core.models import EvaluationData, TurnData
from lightspeed_evaluation.core.models.data import (
    ConversationMetadata,
    DatasetMetadata,
)
from lightspeed_evaluation.core.output.data_persistence import save_evaluation_data


def test_save_evaluation_data_with_path_objects(tmp_path: Path) -> None:
    """Test that Path objects are properly serialized as strings in YAML."""
    # Create test data with Path objects
    setup_script_path = Path("config/sample_scripts/setup.sh")
    cleanup_script_path = Path("config/sample_scripts/cleanup.sh")
    verify_script_path = Path("config/sample_scripts/verify.sh")

    evaluation_data = [
        EvaluationData(
            conversation_group_id="test_conv_1",
            description="Test conversation with script paths",
            setup_script=setup_script_path,
            cleanup_script=cleanup_script_path,
            turns=[
                TurnData(
                    turn_id="turn_1",
                    query="Test query",
                    response="Test response",
                    verify_script=verify_script_path,
                )
            ],
        )
    ]

    # Create temporary evaluation data file
    temp_eval_file = tmp_path / "test_evaluation_data.yaml"
    temp_eval_file.write_text("# Test evaluation data\n")

    # Save evaluation data
    output_file = save_evaluation_data(
        evaluation_data=evaluation_data,
        original_data_path=str(temp_eval_file),
        output_dir=str(tmp_path),
    )

    # Verify output file was created
    assert output_file is not None
    output_path = Path(output_file)
    assert output_path.exists()

    # Load the saved YAML and verify Path objects are strings
    with open(output_path, "r", encoding="utf-8") as f:
        saved_data = yaml.safe_load(f)

    # Verify the structure
    assert len(saved_data) == 1
    conversation = saved_data[0]

    # Verify script paths are strings, not PosixPath objects
    assert isinstance(conversation["setup_script"], str)
    assert isinstance(conversation["cleanup_script"], str)
    assert conversation["setup_script"] == "config/sample_scripts/setup.sh"
    assert conversation["cleanup_script"] == "config/sample_scripts/cleanup.sh"

    # Verify turn script path is also a string
    assert len(conversation["turns"]) == 1
    turn = conversation["turns"][0]
    assert isinstance(turn["verify_script"], str)
    assert turn["verify_script"] == "config/sample_scripts/verify.sh"

    # Verify no PosixPath serialization artifacts in the file
    with open(output_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    assert "!!python/object/apply:pathlib.PosixPath" not in file_content
    assert "PosixPath" not in file_content


def test_save_evaluation_data_with_string_paths(tmp_path: Path) -> None:
    """Test that string paths remain as strings in YAML."""
    # Create test data with string paths
    evaluation_data = [
        EvaluationData(
            conversation_group_id="test_conv_2",
            description="Test conversation with string paths",
            setup_script="sample_scripts/setup.sh",
            cleanup_script="sample_scripts/cleanup.sh",
            turns=[
                TurnData(
                    turn_id="turn_1",
                    query="Test query",
                    response="Test response",
                    verify_script="sample_scripts/verify.sh",
                )
            ],
        )
    ]

    # Create temporary evaluation data file
    temp_eval_file = tmp_path / "test_evaluation_data.yaml"
    temp_eval_file.write_text("# Test evaluation data\n")

    # Save evaluation data
    output_file = save_evaluation_data(
        evaluation_data=evaluation_data,
        original_data_path=str(temp_eval_file),
        output_dir=str(tmp_path),
    )

    # Verify output file was created
    assert output_file is not None
    output_path = Path(output_file)
    assert output_path.exists()

    # Load the saved YAML
    with open(output_path, "r", encoding="utf-8") as f:
        saved_data = yaml.safe_load(f)

    # Verify paths are still strings
    conversation = saved_data[0]
    assert conversation["setup_script"] == "sample_scripts/setup.sh"
    assert conversation["cleanup_script"] == "sample_scripts/cleanup.sh"
    assert conversation["turns"][0]["verify_script"] == "sample_scripts/verify.sh"


def test_save_evaluation_data_without_scripts(tmp_path: Path) -> None:
    """Test that evaluation data without scripts saves correctly."""
    evaluation_data = [
        EvaluationData(
            conversation_group_id="test_conv_3",
            description="Test conversation without scripts",
            turns=[
                TurnData(
                    turn_id="turn_1",
                    query="Test query",
                    response="Test response",
                )
            ],
        )
    ]

    # Create temporary evaluation data file
    temp_eval_file = tmp_path / "test_evaluation_data.yaml"
    temp_eval_file.write_text("# Test evaluation data\n")

    # Save evaluation data
    output_file = save_evaluation_data(
        evaluation_data=evaluation_data,
        original_data_path=str(temp_eval_file),
        output_dir=str(tmp_path),
    )

    # Verify output file was created
    assert output_file is not None
    assert Path(output_file).exists()


def test_save_evaluation_data_error_handling(tmp_path: Path) -> None:
    """Test error handling when saving evaluation data fails."""
    evaluation_data = [
        EvaluationData(
            conversation_group_id="test_conv_4",
            turns=[
                TurnData(
                    turn_id="turn_1",
                    query="Test query",
                )
            ],
        )
    ]

    # Try to save to an invalid path (file instead of directory)
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("Invalid")

    result = save_evaluation_data(
        evaluation_data=evaluation_data,
        original_data_path="test.yaml",
        output_dir=str(invalid_file),  # This is a file, not a directory
    )

    # Should return None on error
    assert result is None


def test_save_evaluation_data_with_dataset_metadata(tmp_path: Path) -> None:
    """Test that dataset metadata is preserved in the dict-format output."""
    dataset_meta = DatasetMetadata(
        team_product="Team LEADS",
        dataset_version="1.0",
        pii_confirmed_removed=True,
        generation_tools=["SDG-hub"],
        llms_used=["gpt-4o"],
        additional_metadata={"grade": "Gold"},
    )
    evaluation_data = [
        EvaluationData(
            conversation_group_id="conv1",
            turns=[TurnData(turn_id="t1", query="Q", response="A")],
        )
    ]

    temp_eval_file = tmp_path / "test_eval.yaml"
    temp_eval_file.write_text("# placeholder\n")

    output_file = save_evaluation_data(
        evaluation_data=evaluation_data,
        original_data_path=str(temp_eval_file),
        output_dir=str(tmp_path),
        dataset_metadata=dataset_meta,
    )

    assert output_file is not None
    with open(output_file, "r", encoding="utf-8") as f:
        saved = yaml.safe_load(f)

    # Should be dict format with metadata + conversations
    assert isinstance(saved, dict)
    assert "metadata" in saved
    assert "conversations" in saved
    assert saved["metadata"]["team_product"] == "Team LEADS"
    assert saved["metadata"]["dataset_version"] == "1.0"
    assert saved["metadata"]["pii_confirmed_removed"] is True
    assert saved["metadata"]["generation_tools"] == ["SDG-hub"]
    assert saved["metadata"]["llms_used"] == ["gpt-4o"]
    assert saved["metadata"]["additional_metadata"] == {"grade": "Gold"}
    assert len(saved["conversations"]) == 1
    assert saved["conversations"][0]["conversation_group_id"] == "conv1"


def test_save_evaluation_data_without_dataset_metadata_uses_list(
    tmp_path: Path,
) -> None:
    """Test that omitting dataset metadata keeps the original list format."""
    evaluation_data = [
        EvaluationData(
            conversation_group_id="conv1",
            turns=[TurnData(turn_id="t1", query="Q", response="A")],
        )
    ]

    temp_eval_file = tmp_path / "test_eval.yaml"
    temp_eval_file.write_text("# placeholder\n")

    output_file = save_evaluation_data(
        evaluation_data=evaluation_data,
        original_data_path=str(temp_eval_file),
        output_dir=str(tmp_path),
    )

    assert output_file is not None
    with open(output_file, "r", encoding="utf-8") as f:
        saved = yaml.safe_load(f)

    assert isinstance(saved, list)
    assert len(saved) == 1


def test_save_preserves_conversation_metadata(tmp_path: Path) -> None:
    """Test that conversation-level metadata round-trips via save."""
    evaluation_data = [
        EvaluationData(
            conversation_group_id="conv1",
            metadata=ConversationMetadata(
                scenario_category="Edge Case",
                use_case="RAG",
                complexity="Complex",
                human_verified=True,
                additional_metadata={"priority": "high"},
            ),
            turns=[
                TurnData(
                    turn_id="t1",
                    query="Q",
                    response="A",
                )
            ],
        )
    ]

    temp_eval_file = tmp_path / "test_eval.yaml"
    temp_eval_file.write_text("# placeholder\n")

    output_file = save_evaluation_data(
        evaluation_data=evaluation_data,
        original_data_path=str(temp_eval_file),
        output_dir=str(tmp_path),
    )

    assert output_file is not None
    with open(output_file, "r", encoding="utf-8") as f:
        saved = yaml.safe_load(f)

    conv = saved[0]
    assert conv["metadata"]["scenario_category"] == "Edge Case"
    assert conv["metadata"]["use_case"] == "RAG"
    assert conv["metadata"]["complexity"] == "Complex"
    assert conv["metadata"]["human_verified"] is True
    assert conv["metadata"]["additional_metadata"] == {"priority": "high"}
