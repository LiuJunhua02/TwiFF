
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import os
import json
from datetime import datetime

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

import random
import numpy as np

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
import argparse

COT_SYSTEM_PROMPT = '''You are an AI assistant capable of reasoning with visual imagery. You should conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step with image. After fully reasoning through the problem—potentially using image-based thinking—provide only a clear, concise, and direct answer to the user's question.
'''.strip()

# Rest of inference code
from inferencer import InterleaveInferencer

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
    parser.add_argument('--checkpoint_dir', default='/path/to/TwiFF/checkpoints', type=str)
    parser.add_argument('--QA_file', default='/path/to/QA.jsonl', type=str)
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    seed = args.seed
    set_seed(seed)
    model_dir = args.model_dir
    checkpoint_file = args.checkpoint_file
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"GPU memory per device:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

    # LLM config preparing (use base model configs)
    llm_config = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing (use base model configs)
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading (use base model VAE)
    vae_model, vae_config = load_ae(local_path=os.path.join(model_dir, "ae.safetensors"))
    vae_model = vae_model.to(torch.bfloat16).to('cuda').eval()

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # Create model with empty weights - IMPORTANT: Use float32 initially to match checkpoint
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer Preparing (use base model tokenizer)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing for cot gen
    vae_transform = ImageTransform(512, 256, 16)
    vit_transform = ImageTransform(518, 224, 14)

    # Device mapping for 8x80GB GPUs - use bf16 directly
    max_mem_per_gpu = "80GiB"

    print("Setting up device mapping...")
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        dtype=torch.bfloat16,  # Use bf16 for device mapping
    )

    print("Device map:", device_map)

    # Handle same-device modules
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        if first_device is not None:
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

    print("Final device map:", device_map)

    # Load checkpoint directly in bf16
    print(f"Loading checkpoint directly in bfloat16: {checkpoint_path}")
    print("Loading model from safetensors file...")

    # Load model directly in bf16
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=False,
        dtype=torch.bfloat16,   # Load directly as bf16
        force_hooks=True,
    )

    model = model.eval()

    print('Model loaded directly in bfloat16!')
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print("Model loading completed successfully!")

    # Check memory usage
    print("GPU memory usage after loading:")
    for i in range(torch.cuda.device_count()):
        if torch.cuda.memory_allocated(i) > 0:
            allocated = torch.cuda.memory_allocated(i) / 1e9
            cached = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached")

    # data setting
    with open(args.QA_file, 'r') as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]

    # Create output folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"./output/reasoning_output_{timestamp}"
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    problem_image_paths = []
    
    # Save JSON file
    json_path = os.path.join(output_folder, "reasoning_data.jsonl")
    #generate the answer
    for q_cnt, ditem in enumerate(items):
        print('init the inferencer!')
        inferencer = InterleaveInferencer(
            model=model, 
            vae_model=vae_model, 
            tokenizer=tokenizer, 
            vae_transform=vae_transform, 
            vit_transform=vit_transform, 
            new_token_ids=new_token_ids
        )
        inference_hyper=dict(
            do_sample=True,
            text_temperature=0.3,
            cfg_text_scale=3.5,
            cfg_img_scale=2.0,
            cfg_interval=[0.0, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="channel",
        )
        prompt = ditem['prompt']
        image = []
        for img_path in ditem['img_path']:
            image.append(Image.open(img_path))
        print('*'*50)
        print(prompt)
        print('-'*50)
        assert prompt.count('<image>') == len(image), f'question_{q_cnt} missing the image placeholder!'

        # Handle the input and save the original images
        current_input = []
        if image is not None:
            for i, img in enumerate(image):
                problem_image_path = os.path.join(images_folder, f"problem_{q_cnt}_image_{i+1}.png")
                relative_path = os.path.join("images", f"problem_{q_cnt}_image_{i+1}.png")
                img.save(problem_image_path)
                problem_image_paths.append(relative_path)
                print(f"Problem {q_cnt} image {i+1} saved at '{problem_image_path}'")
                #prepare the question
                pre_parts, prompt = prompt.split('<image>', 1)
                if pre_parts.strip():
                    current_input.append(pre_parts)
                current_input.append(img)
            if prompt.strip():
                current_input.append(prompt)
        else:
            current_input = [prompt]

        #test cot gen
        output = inferencer.cot_inference(current_input, system_prompt=COT_SYSTEM_PROMPT, max_round=args.max_round, **inference_hyper)
        reasoning_data = ""
        reasoning_image_paths = []
        for iteration, item in enumerate(output):
            if isinstance(item, str):
                print(item)
                reasoning_data+=item
            elif isinstance(item, Image.Image):
                reasoning_data+='<rimage>'
                image_filename = f'reasoning_{q_cnt}_image_{iteration + 1}.png'
                image_path = os.path.join(images_folder, image_filename)
                reasoning_image_paths.append(image_path)
                item.save(image_path)
            else:
                print('Unknow output')
                continue
        result_dict = {
            "prompt": ditem['prompt'],
            "system_prompt": COT_SYSTEM_PROMPT,
            "problem_image_paths": problem_image_paths if problem_image_paths else None,
            "response": reasoning_data,
            "resoning_image_paths": reasoning_image_paths if reasoning_image_paths else None,
        }
        with open(json_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
print(f"\nReasoning complete!")
print(f"Output folder: {output_folder}")
print(f"JSON metadata: {json_path}")

