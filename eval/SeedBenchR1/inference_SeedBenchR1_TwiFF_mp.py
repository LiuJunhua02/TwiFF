
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent.parent))
import os
import json
from datetime import datetime
from copy import deepcopy
from decord import VideoReader
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

from PIL import Image
import torch

import random
import numpy as np

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from safetensors.torch import load_file
from modeling.autoencoder import load_ae
import argparse
from eval.TwiFFBench.inferencer_batch import InterleaveInferencer

COT_SYSTEM_PROMPT = '''You are an AI assistant capable of reasoning with visual imagery. You should conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step with image. After fully reasoning through the problem—potentially using image-based thinking—provide only a clear, concise, and direct answer to the user's question.
'''.strip()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_round', default=2, type=int)
    parser.add_argument('--model_dir', default='/path/to/BAGEL-7B-MoT', type=str)
    parser.add_argument('--checkpoint_file', default='model.safetensors', type=str)
    parser.add_argument('--checkpoint_dir', default='/path/to/checkpoints/', type=str)
    parser.add_argument('--QA_file', default='/path/to/benchmark', type=str)
    parser.add_argument('--data_root_dir', default='/path/to/benchmark_image', type=str)
    parser.add_argument('--output_dir', default='/path/to/output_dir', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--visual_gen', action='store_true', help='Enable visual generation')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--rank', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--disable_options', action='store_true', help='disable options')
    return parser.parse_args()

def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, start_frames=0, end_frames=0, clip=None
):
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()

    t_num_frames = num_frames

    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, start_frames=start_frames, end_frames=end_frames,
    )
    if clip is not None:
        tmp = [frame_indices[i - 1] for i in clip]
        frame_indices = tmp
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1, 
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


def setup_model(model_dir, checkpoint_path, device):

    # --- Configs ---
    llm_config = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_dir, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=args.visual_gen,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # --- Build full model on current device ---
    print(f"[Rank {rank}] Building model...")
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=False)

    # --- Load weights directly to device ---
    print(f"[Rank {rank}] Loading checkpoint...")
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path, device='cpu')
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(state_dict, strict=True)
    model.to(torch.bfloat16).to(device).eval()

    # --- Tokenizer & transforms ---
    tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    # Set up transforms
    vae_transform = ImageTransform(512, 256, 16)
    vit_transform = ImageTransform(518, 224, 14)

    vae_model = vae_model.to(torch.bfloat16).to(device).eval()

    return model, vae_model, vae_transform, vit_transform, new_token_ids, tokenizer

if __name__ == '__main__':
    args = get_args()
    data_root_dir = args.data_root_dir
    device = torch.device("cuda:0")
    rank = args.rank
    world_size = args.world_size

    with open(args.QA_file, 'r') as f:
        dataset = [json.loads(line.strip()) for line in f]

    props = torch.cuda.get_device_properties(device)
    print(f"[Rank {rank}] GPU: {props.name}, Memory: {props.total_memory / 1e9:.1f} GB")

    seed = args.seed
    set_seed(seed)

    model_dir = args.model_dir
    checkpoint_file = args.checkpoint_file
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    model, vae_model, vae_transform, vit_transform, new_token_ids, tokenizer = setup_model(model_dir, checkpoint_path, device)
    print(f"[Rank {rank}] Model loaded in bfloat16! Dtype: {next(model.parameters()).dtype}")

    # Data loading
    total_items = len(dataset)
    items_per_gpu = (total_items + world_size - 1) // world_size
    start = rank * items_per_gpu
    end = min(start + items_per_gpu, total_items)


    output_folder = args.output_dir
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    rank_json_path = os.path.join(output_folder, f"model_response_rank{rank}.jsonl")

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        device=device
    )

    inference_hyper = dict(
        do_sample=True,
        text_temperature=0.3,
        cfg_text_scale=3.5,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="channel",
        enable_taylorseer=False,
        use_tqdm=False,
        und_only=not args.visual_gen,
        max_think_token_n=4096,
    )
    print(f"[Rank {rank}]: Processing indices {start} to {end - 1} ({end - start} items)")
    chosen_list = ['A','B',"C","D","E","F"]
    chosen_dict = {'A': 'choice_a', 'B': 'choice_b', 'C': 'choice_c', 'D': 'choice_d', 'E': 'choice_e', 'F': 'choice_f'}
    for idx in range(start, end):
        ditem = dataset[idx]
        result_dict = deepcopy(ditem)
        index = ditem['sample_id']
        prompt = ditem['question']
        prompt = prompt.replace('Considering my current observation shown in the image, ', '')
        prompt = prompt.replace('Considering the progress shown in the video and my current observation shown in the image, ', '')
        prompt = 'frame_1\n<image>\n'+prompt.strip()

        answer = ditem['answer']
        if args.disable_options:
            pass
        else:
            prompt += '\nOptions:'
            for opt_num in chosen_list:
                opt_val = ditem.get(chosen_dict[opt_num],None)
                if (opt_val is not None) and (opt_val!='nan'):
                    prompt+=f'\n{opt_num}. {opt_val}'
        video_frames=[]
        
        image_path = os.path.join(data_root_dir, 'images', ditem['video_source'], ditem['current_observation_basename'])
        img = Image.open(image_path)
        img = img.convert('RGB')
        image = [img]
        current_input = []
        video_input_list = []
        if(len(video_frames)>0):
            for frame in video_frames:
                video_input_list.append([frame])
        if image is not None:
            for i, img in enumerate(image):
                pre_parts, prompt = prompt.split('<image>', 1)
                if pre_parts.strip():
                    current_input.append([pre_parts])
                current_input.append([img])
            if prompt.strip():
                current_input.append([prompt])
        else:
            current_input = [[prompt]]

        output, is_end_list = inferencer.cot_inference(current_input, system_prompt=COT_SYSTEM_PROMPT, max_round=args.max_round, video_input_list=video_input_list, **inference_hyper)

        all_results = []
        for k in range(len(is_end_list)):
            reasoning_data = []
            for iteration, item in enumerate(output):
                if isinstance(item[k], str):
                    reasoning_data.append(item[k])
                elif isinstance(item[k], Image.Image):
                    image_filename = f'reasoning_{index}_image_{iteration + 1}.png'
                    image_path = os.path.join(images_folder, image_filename)
                    reasoning_data.append(image_path)
                    item[k].save(image_path)
                else:
                    print('Unknown output type')
                    continue
            result_dict['model_response'] = reasoning_data
            all_results.append(result_dict)
        with open(rank_json_path, 'a', encoding='utf-8') as f:
            for res in all_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        if (idx - start + 1) % 5 == 0:
            timestamp_save = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"[Rank {rank}] [{timestamp_save}] Processed {idx - start + 1}/{end - start} items")
    timestamp_save = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[Rank {rank}] [{timestamp_save}] Done!")

