import json
import os
import re
import traceback
from PIL import Image, ImageFile, PngImagePlugin

import random
import numpy as np
import decord

from .interleave_t2i_dataset import InterleavedBaseIterableDataset
from ..data_utils import pil_img2rgb
from ..distributed_iterable_dataset import DistributedIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

ICOT_PROMPT = '''
You are an AI assistant capable of reasoning with visual imagery. You should conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step with image. After fully reasoning through the problem—potentially using image-based thinking—provide only a clear, concise, and direct answer to the user's question.
'''.strip()


class InterleaveReasoningIterableDataset(InterleavedBaseIterableDataset, DistributedIterableDataset):
    def __init__(
        self, 
        dataset_name, 
        transform, 
        tokenizer, 
        vit_transform,
        jsonl_path_list, 
        data_dir_list, 
        num_used_data,
        local_rank=0, 
        world_size=1, 
        num_workers=8, 
        data_status=None,
        shuffle_lines=True, 
        shuffle_seed=0,
        image_prefix_dir=None,
    ):
        """
        Dataset for think-trace style JSONL files with interleaved text and images.
        
        Args:
            dataset_name: Name of the dataset
            transform: Transform for VAE images  
            tokenizer: Text tokenizer
            vit_transform: Transform for VIT images
            jsonl_path_list: List of JSONL file paths
            data_dir_list: List of base directories (should match jsonl_path_list)
            num_used_data: List of number of samples to use from each JSONL. If a value is None or non-positive, all data from that JSONL will be used.
            image_prefix_dir: Absolute path to prepend to relative image paths
            Other args: Standard distributed dataset args
        """
        DistributedIterableDataset.__init__(self, dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform  
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.image_prefix_dir = image_prefix_dir or ""
        
        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            num_used_data, 
            shuffle_lines,
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(self, jsonl_path_list, num_used_data, shuffle_lines, shuffle_seed):
        data_paths = []
        if not isinstance(num_used_data, list):
            num_used_data = [num_used_data] * len(jsonl_path_list)

        for jsonl_path, num_data_point in zip(jsonl_path_list, num_used_data):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            
            # Convert 'None' string to None type
            if num_data_point == 'None':
                num_data_point = None

            if num_data_point is not None and int(num_data_point) > 0:
                raw_data = raw_data[:int(num_data_point)]

            data_paths.extend(raw_data)
        return data_paths
    
    def get_answer(self, text):
        match = re.match(r'(.*?)<ans>.*?</ans>', text)
        if match:
            prefix = match.group(1)                     # first part, reasoning
            rest = text[len(prefix):]                      # second part, answer
            return prefix, rest
        else:
            #no answer find
            return text, ''

    def parse_row(self, json_line):
        """Parse a single JSON line into the required format"""
        tmp_image = []
        try:
            data_item = json.loads(json_line.strip())
            for img_path in data_item['images']:
                img_path = os.path.join(self.image_prefix_dir, img_path)
                tmp_image.append(pil_img2rgb(Image.open(img_path)))
            num_input_images = 1
            num_output_images = 1
            # Build the sequence
            data = self._init_data()
        except:
            traceback.print_exc()
            return {}
        if(len(tmp_image) != num_input_images+num_output_images):
            print(f"Unmatch images! extract: {len(tmp_image)}, images excepted: {num_input_images+num_output_images}")
            return {}
        image_index = 0

        enable_cfg_flg = num_output_images > 0#For text only cot we will not use cfg
        last_conv = False

        #add the system prompt for missing system prompts
        if data_item['conversations'][0]['from']!='system':
            sys_conv = {"from": "system", "value": f"{ICOT_PROMPT}"}
            data_item['conversations'].insert(0, sys_conv)

        for i, conversation in enumerate(data_item['conversations']):
            if i == len(data_item['conversations']) - 1:
                last_conv = True
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                prompt_parts = conversation['value']
                for i in range(image_cnt):
                    if image_index == num_input_images:
                        break
                    pre_parts, prompt_parts = prompt_parts.split('<image>', 1)
                    if pre_parts.strip():
                        data = self._add_text(data, pre_parts.strip(), need_loss=False, enable_cfg=enable_cfg_flg)
                    data = self._add_image(
                        data, tmp_image[image_index], 
                        need_loss=False, # No loss for question images
                        need_vae=True,   # VAE conditioning
                        need_vit=True,   # VIT understanding
                        enable_cfg=enable_cfg_flg and (image_index<num_input_images+num_output_images-1), 
                    )
                    image_index=1+image_index
                if(prompt_parts.strip()):
                    data = self._add_text(data, prompt_parts.strip(), need_loss=False, enable_cfg=enable_cfg_flg)
            elif conversation['from'] == 'gpt':
                # prompt_parts = conversation['value']+'\n<ansend>'#to stop the generate loop.
                prompt_parts = conversation['value']
                image_cnt = conversation['value'].count('<rimage>')
                for i in range(image_cnt):
                    if image_index == num_input_images+num_output_images:
                        break
                    pre_parts, prompt_parts = prompt_parts.split('<rimage>', 1)
                    if pre_parts.strip():
                        data = self._add_text(data, pre_parts.strip(), need_loss=True, enable_cfg=enable_cfg_flg)
                    data = self._add_image(
                        data, tmp_image[image_index], 
                        need_loss=True, # img generation loss
                        need_vae=True,   # VAE conditioning
                        need_vit=True,   # VIT understanding
                        enable_cfg=enable_cfg_flg and (image_index<num_input_images+num_output_images-1),
                    )
                    image_index=1+image_index
                if(prompt_parts.strip()):
                    if last_conv:
                        data = self._add_text(data, prompt_parts.strip(), need_loss=True, enable_cfg=False)
                        # reasoning_part, answer_part = self.get_answer(prompt_parts)
                        # if(reasoning_part.strip()):
                        #     data = self._add_text(data, reasoning_part.strip(), need_loss=True, enable_cfg=enable_cfg_flg)
                        # if(answer_part.strip()):
                        #     # final answer will disable cfg drop
                        #     data = self._add_text(data, answer_part.strip(), need_loss=True, enable_cfg=False)
                    else:
                        data = self._add_text(data, prompt_parts.strip(), need_loss=True, enable_cfg=enable_cfg_flg)
            elif conversation['from'] == 'system':
                self._add_text(data, conversation['value'], need_loss=False, enable_cfg=False)# sys will not drop
            else:
                role = conversation['from']
                print(f'Unknow conv role:{role}')
                return {}
        if(image_index != num_input_images+num_output_images):
            print(f"Mismatch in question: {image_index=}, images: {num_input_images+num_output_images}")
            return {}

        return data

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )
        repeat = 0
        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            self.rng.seed(42+repeat+worker_id+self.local_rank)
            self.rng.shuffle(data_paths_per_worker_)
            for row_idx, json_line in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data = self.parse_row(json_line)
                    if len(data) == 0:
                        continue

                    # Check if we have any loss 
                    has_loss = any(item['loss'] for item in data['sequence_plan'])
                    if not has_loss:
                        print('No loss defined, skipped.')
                        continue

                    data['data_indexes'] = {
                        "data_indexes": row_idx,
                        "worker_id": worker_id, 
                        "dataset_name": self.dataset_name,
                    }
                    yield data

                except Exception as e:
                    print(f"Error processing row {row_idx}: {e}")
                    traceback.print_exc()
                    continue

            row_start_id = 0
            repeat+=1
            print(f"{self.dataset_name} repeat#{repeat} in rank-{self.local_rank} worker-{worker_id}")