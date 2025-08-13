#!/usr/bin/env python3
"""
Simple test to verify the multi_record function and configurations work correctly.
"""

from lerobot.record import DatasetRecordConfig, MultiDatasetRecordConfig


def test_multi_dataset_config():
    """Test that MultiDatasetRecordConfig creates correctly."""

    # Create test dataset configurations
    dataset1 = DatasetRecordConfig(repo_id="test/dataset1", single_task="Test task 1")

    dataset2 = DatasetRecordConfig(repo_id="test/dataset2", single_task="Test task 2")

    # Test multi-dataset config creation
    multi_config = MultiDatasetRecordConfig(datasets=[dataset1, dataset2], use_numeric_keys=True)

    assert len(multi_config.datasets) == 2
    assert multi_config.use_numeric_keys == True
    print("✓ MultiDatasetRecordConfig creation test passed")


def test_multi_dataset_config_defaults():
    """Test that default use_numeric_keys is True."""

    dataset1 = DatasetRecordConfig(repo_id="test/dataset1", single_task="Test task 1")
    dataset2 = DatasetRecordConfig(repo_id="test/dataset2", single_task="Test task 2")
    dataset3 = DatasetRecordConfig(repo_id="test/dataset3", single_task="Test task 3")

    # Test with default setting
    multi_config = MultiDatasetRecordConfig(datasets=[dataset1, dataset2, dataset3])

    assert multi_config.use_numeric_keys == True  # Default should be True
    print("✓ Default use_numeric_keys test passed")


def test_validation():
    """Test configuration validation."""

    # Test empty datasets list
    try:
        MultiDatasetRecordConfig(datasets=[])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "At least one dataset configuration must be provided" in str(e)
        print("✓ Empty datasets validation test passed")

    # Test too many datasets (more than 9)
    datasets = []
    for i in range(10):
        datasets.append(DatasetRecordConfig(repo_id=f"test/dataset{i}", single_task=f"Test task {i}"))

    try:
        MultiDatasetRecordConfig(datasets=datasets, use_numeric_keys=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Maximum of 9 datasets" in str(e)
        print("✓ Too many datasets validation test passed")


if __name__ == "__main__":
    print("Running multi-dataset recording configuration tests...")

    test_multi_dataset_config()
    test_multi_dataset_config_defaults()
    test_validation()

    print("\n✅ All tests passed! Multi-dataset recording configuration is working correctly.")