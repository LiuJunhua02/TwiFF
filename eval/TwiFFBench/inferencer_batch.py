# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any, Tuple

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache

def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, device):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.device = device
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, texts, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=texts,
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        generation_input = move_generation_input_to_device(generation_input, device=self.device)
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, images, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=images,
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            generation_input = move_generation_input_to_device(generation_input, device=self.device)
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=images,
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            generation_input = move_generation_input_to_device(generation_input, device=self.device)
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
        use_tqdm=False,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        generation_input = move_generation_input_to_device(generation_input, device=self.device)
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device=self.device)
        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device=self.device)
        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
            use_tqdm=use_tqdm,
        )
        images = [self.decode_image(latent, image_shape) for latent in unpacked_latent]
        # image = self.decode_image(unpacked_latent[0], image_shape)
        return images

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        generation_input = move_generation_input_to_device(generation_input, device=self.device)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        outputs = self.tokenizer.batch_decode(unpacked_latent.t(), skip_special_tokens=False)
        text_outputs = []
        for i in range(len(outputs)):
            text_outputs.append(outputs[i].split('<|im_end|>')[0].split('<|im_start|>')[1])
        # output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return text_outputs
    

    @torch.no_grad()
    def cot_inference(
        self,
        input_lists: List[Union[List[str], List[Image.Image]]],
        video_input_list=None,
        system_prompt=None,
        max_round = 5,#max think round

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(512, 512),
        enable_taylorseer=False,
        use_tqdm=False,
        und_only=False,
    ) -> Tuple[List[Union[List[str], List[Image.Image]]], List[int]]:
        """
        input_lists: [[Text_1_1, Text_2_1], [Image1_1, Image2_1], [Text1_2, Text2_2]]

        """
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        micro_batch_size = len(input_lists[0])
        if cfg_renorm_type=="global" and micro_batch_size>1:
            raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted for batch>1 inference")

        is_end_list = [-1]*micro_batch_size
        is_end_cnt = 0
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if system_prompt is not None:
                gen_context = self.update_context_text([system_prompt]*micro_batch_size, gen_context)
                cfg_img_context = deepcopy(gen_context)
            if (video_input_list is not None) and len(video_input_list)>0:
                for input_term in video_input_list:
                    if isinstance(input_term[0], Image.Image):
                        input_term = [self.vae_transform.resize_transform(pil_img2rgb(item)) for item in input_term]
                        gen_context = self.update_context_image(input_term, gen_context, vae=False)
                        cfg_text_context = deepcopy(gen_context)
                        image_shapes = input_term[0].size[::-1]

                    else:
                        raise ValueError(f"Unsupported input type: {type(input_term[0])}")

            for input_term in input_lists:
                if isinstance(input_term[0], str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term[0], Image.Image):
                    input_term = [self.vae_transform.resize_transform(pil_img2rgb(item)) for item in input_term]
                    gen_context = self.update_context_image(input_term, gen_context, vae=not und_only)
                    cfg_text_context = deepcopy(gen_context)
                    image_shapes = input_term[0].size[::-1]

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term[0])}")

            curr_round = 0
            while curr_round<max_round:
                gen_texts = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_texts)
                curr_index = len(output_list)-1

                # gen_img_flag = False
                for i,text in enumerate(gen_texts):
                    if (('</ans>' in text or '<ans>' in text) or und_only) and is_end_list[i]==-1:#got the final answer
                        is_end_list[i]=curr_index
                        is_end_cnt+=1
                    if ('</answer>' in text or '<answer>' in text) and is_end_list[i]==-1:
                        is_end_list[i]=curr_index
                        is_end_cnt+=1

                if is_end_cnt==micro_batch_size:
                    break
                gen_context = self.update_context_text(gen_texts, gen_context)
                cfg_img_context = self.update_context_text(gen_texts, cfg_img_context)
                # if gen_img_flag:
                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                    use_tqdm=use_tqdm,
                )
                cfg_img_context = deepcopy(gen_context)
                gen_context = self.update_context_image(img, gen_context, vae=True)
                cfg_text_context = deepcopy(gen_context)
                output_list.append(img)
                curr_round=curr_round+1
            if curr_round>=max_round:
                print("Think out of the round limit!")
        return output_list, is_end_list