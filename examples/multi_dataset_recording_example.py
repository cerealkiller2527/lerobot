#!/usr/bin/env python3
"""
Example script for multi-dataset recording using the new multi_record function.

This example shows how to record data for multiple datasets sequentially within the same episode.
For instance, you can record "pick" and "place" motions as separate datasets but within the same
continuous episode.

The new system uses numeric keys (1-9) for direct stage switching:
- Press '1' to switch to the first dataset
- Press '2' to switch to the second dataset
- Press '3' to switch to the third dataset
- etc.

Usage:
```shell
python examples/multi_dataset_recording_example.py
```

Or run the multi_record function directly:
```shell
python -m lerobot.record --config=examples/multi_dataset_recording_example.py
```
"""

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.kinect.configuration_kinect import KinectCameraConfig
from lerobot.record import DatasetRecordConfig, MultiDatasetRecordConfig, MultiRecordConfig
from lerobot.robots.bi_so101_follower import BiSO101FollowerConfig
from lerobot.teleoperators.bi_so101_leader import BiSO101LeaderConfig


def create_multi_record_config():
    """Create a configuration for multi-dataset recording."""
    
    # Try to get HuggingFace username automatically
    import subprocess
    try:
        result = subprocess.run(["huggingface-cli", "whoami"], capture_output=True, text=True)
        if result.returncode == 0:
            HF_USER = result.stdout.strip().split('\n')[0]
            print(f"Using HuggingFace username: {HF_USER}")
        else:
            HF_USER = "cerealkiller2527"  # Default username, change this to your HuggingFace username
            print(f"Warning: Not logged into HuggingFace. Using default username: {HF_USER}")
            print("To use your own username, run: huggingface-cli login")
    except Exception:
        HF_USER = "cerealkiller2527"  # Default username
        print(f"Using default username: {HF_USER}")

    # Define the robot configuration (bi-SO101 with 2 RealSense + 1 Kinect)
    robot_config = BiSO101FollowerConfig(
        left_arm_port="COM6",
        right_arm_port="COM3",
        cameras={
            "cam_low": RealSenseCameraConfig(
                serial_number_or_name="218622270973",
                width=848,
                height=480,
                fps=30,
            ),
            "cam_high": RealSenseCameraConfig(
                serial_number_or_name="218622278797",
                width=848,
                height=480,
                fps=30,
            ),
            "cam_kinect": KinectCameraConfig(
                device_index=0,
                width=848,
                height=480,
                fps=30,
            ),
        },
        id="my_bimanual",
    )

    # Define teleoperator configuration for bi-manual control
    teleop_config = BiSO101LeaderConfig(
        left_arm_port="COM5",
        right_arm_port="COM4",
        id="my_bimanual_leader",
    )

    episodes = 1  # Number of episodes should be the same for both datasets
    reset_time = 5
    episode_time = 30
    private_repo = False
    push_hub = True
    batch_size = 10  # Encode videos every 10 episodes for better performance
    
    # Define multiple dataset configurations for different stages
    dataset_configs = [
        DatasetRecordConfig(
            repo_id=f"{HF_USER}/pick_knife",
            single_task="Pick up the knife from the table.",
            fps=30,
            episode_time_s=episode_time,
            reset_time_s=reset_time,  # Increased reset time for multi-camera setup
            num_episodes=episodes,
            video=True,
            push_to_hub=push_hub,  # Set to True if you want to upload
            private=private_repo,
            num_image_writer_threads_per_camera=6,  # Optimized for multi-camera
            video_encoding_batch_size=batch_size,  # Batch encode every 10 episodes
        ),
        DatasetRecordConfig(
            repo_id=f"{HF_USER}/place_left_knife",
            single_task="Place the knife to the left of the plate.",
            fps=30,
            episode_time_s=episode_time,
            reset_time_s=reset_time,
            num_episodes=episodes,
            video=True,
            push_to_hub=push_hub,
            private=private_repo,
            num_image_writer_threads_per_camera=6,
            video_encoding_batch_size=batch_size,  # Batch encode every 10 episodes
        ),
        DatasetRecordConfig(
            repo_id=f"{HF_USER}/place_right_knife",
            single_task="Place the knife to the right of the plate.",
            fps=30,
            episode_time_s=episode_time,
            reset_time_s=reset_time,
            num_episodes=episodes,
            video=True,
            push_to_hub=push_hub,
            private=private_repo,
            num_image_writer_threads_per_camera=6,
            video_encoding_batch_size=batch_size,  # Batch encode every 10 episodes
        ),
        DatasetRecordConfig(
            repo_id=f"{HF_USER}/place_inside_knife",
            single_task="Place the knife inside the plate.",
            fps=30,
            episode_time_s=episode_time,
            reset_time_s=reset_time,
            num_episodes=episodes,
            video=True,
            push_to_hub=push_hub,
            private=private_repo,
            num_image_writer_threads_per_camera=6,
            video_encoding_batch_size=batch_size,  # Batch encode every 10 episodes
        ),
    ]

    # Define multi-dataset configuration
    multi_dataset_config = MultiDatasetRecordConfig(
        datasets=dataset_configs,
        use_numeric_keys=True,  # Use numeric keys 1-4 for stage switching
    )

    # Create the complete multi-record configuration
    config = MultiRecordConfig(
        robot=robot_config,
        multi_dataset=multi_dataset_config,
        teleop=teleop_config,
        policy=None,  # No policy, using teleop
        display_data=False,  # Set to False for multi-camera performance
        play_sounds=True,
        resume=False,
    )

    return config


def main():
    """Example of how to use multi_record function."""
    from lerobot.record import multi_record

    # Create configuration
    config = create_multi_record_config()

    print("Starting multi-dataset recording with bi-SO101 robot...")
    print("Configuration: 2 RealSense RGB (848x480) + 1 Kinect RGB (848x480) @ 30 FPS")
    print("Video Encoding: Batched every 10 episodes for optimal performance")
    print()
    print("Instructions:")
    print("- Press '1' to record 'pick knife' motion to the first dataset")
    print("- Press '2' to record 'place left' motion to the second dataset")
    print("- Press '3' to record 'place right' motion to the third dataset")
    print("- Press '4' to record 'place inside' motion to the fourth dataset")
    print("- Press RIGHT ARROW to finish current episode")
    print("- Press LEFT ARROW to re-record current episode")
    print("- Press ESC to stop recording completely")
    print()

    # Start multi-dataset recording
    datasets = multi_record(config)

    print(f"Recording completed! Created {len(datasets)} datasets:")
    for i, dataset in enumerate(datasets):
        print(f"  Dataset {i}: {dataset.repo_id} with {dataset.num_episodes} episodes")


if __name__ == "__main__":
    main()