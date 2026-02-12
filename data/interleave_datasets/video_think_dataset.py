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



def read_frames_decord_uni(
        video_path, num_frames, sample='middle', fix_start=None, start_frames=1, end_frames=1, clip=None
):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()

    t_num_frames = num_frames

    frame_indices = get_frame_indices_uni(
        t_num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, start_frames=start_frames, end_frames=end_frames,
    )
    if clip is not None:
        tmp = [frame_indices[i - 1] for i in clip]
        frame_indices = tmp
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames

def get_frame_indices_uni(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1, 
                      start_frames=0, end_frames=0):
    if sample in ['rand', 'middle']: # uniform sampling
        if start_frames+1>=vlen:
            start_frames = 0
        if vlen-end_frames-1<=start_frames:
            end_frames = vlen-1
        else:
            end_frames=vlen-end_frames-1
        acc_samples = min(num_frames-2, end_frames-start_frames)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=start_frames+1, stop=end_frames, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError
        frame_indices = [start_frames]+frame_indices+[end_frames]
        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    else:
        raise ValueError
    return frame_indices


class VideoThinkJSONLIterableDataset(InterleavedBaseIterableDataset, DistributedIterableDataset):
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

        multi_frame_input_rate=0.0,
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
            multi_frame_input_rate: The rate of input frames (if has) before the question images to format a video input. 
            Other args: Standard distributed dataset args
        """
        DistributedIterableDataset.__init__(self, dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform  
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.image_prefix_dir = image_prefix_dir or ""

        self.multi_frame_input_rate = multi_frame_input_rate
        
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
        try:
            data_item = json.loads(json_line.strip())
            if (random.random() < self.multi_frame_input_rate) and data_item["frames"][0]>1:
                video_input_flag = True
                video_pre=list(range(1, data_item["frames"][0]))
            else:
                video_input_flag = False
                video_pre=[]
            video_path = os.path.join(self.image_prefix_dir, data_item['video'])# only one video
            tmp_image = read_frames_decord_uni(
                    video_path = video_path,
                    num_frames = 8,
                    start_frames= 1,
                    end_frames = 1,
                    sample = 'middle',
                    clip = video_pre + data_item["frames"]+data_item["recon_frames"]
                )
            num_input_images = len(data_item["frames"])
            num_output_images = len(data_item["recon_frames"])
            video_image = tmp_image[:len(video_pre)]
            tmp_image = tmp_image[len(video_pre):]
            # Build the sequence
            data = self._init_data()
        except:
            traceback.print_exc()
            return {}
        if(len(tmp_image) != num_input_images+num_output_images):
            print(f"Unmatch frames! extract: {len(tmp_image)}, frames excepted: {num_input_images+num_output_images}")
            return {}
        image_index = 0

        enable_cfg_flg = num_output_images > 0#For text only cot we will not use cfg
        last_conv = False
        for i, conversation in enumerate(data_item['conversations']):
            if i == len(data_item['conversations']) - 1:
                last_conv = True
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                prompt_parts = conversation['value']
                #insert video frames
                if video_input_flag:
                    for v_img in video_image:
                        data = self._add_image(
                            data, v_img, 
                            need_loss=False, # No loss for video images
                            need_vae=False,   # No VAE for video input
                            need_vit=True,   # VIT understanding
                            enable_cfg=False, 
                        )

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
                    #add image start and end token
                    # pre_parts = pre_parts+'<img>'
                    # prompt_parts = '</img>'+prompt_parts
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
                    #     reasoning_part, answer_part = self.get_answer(prompt_parts)
                    #     if(reasoning_part.strip()):
                    #         data = self._add_text(data, reasoning_part.strip(), need_loss=True, enable_cfg=enable_cfg_flg)
                    #     if(answer_part.strip()):
                    #         # final answer will disable cfg drop
                    #         data = self._add_text(data.strip(), answer_part, need_loss=True, enable_cfg=False)
                    else:
                        data = self._add_text(data, prompt_parts.strip(), need_loss=True, enable_cfg=enable_cfg_flg)
            elif conversation['from'] == 'system':
                self._add_text(data, conversation['value'], need_loss=False, enable_cfg=False)# sys will not drop
            else:
                role = conversation['from']
                print(f'Unknow conv role:{role}')
                return {}
        if(image_index != num_input_images+num_output_images):
            print(f"Mismatch in question: {image_index=}, frames: {num_input_images+num_output_images}")
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