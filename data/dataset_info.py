# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import VideoThinkJSONLIterableDataset


DATASET_REGISTRY = {
    'VideoCOT': VideoThinkJSONLIterableDataset,
}


DATASET_INFO = {
    'VideoCOT': {
        'TwiFF-2_7M': {
            'image_prefix_dir': '/path/to/video',  # Base path for video
		},
    },
}
