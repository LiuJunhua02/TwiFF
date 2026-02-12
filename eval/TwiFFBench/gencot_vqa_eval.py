import json
import asyncio
import aiofiles
from typing import AsyncGenerator, Dict, Any, Set, List
import os
from openai import AsyncOpenAI
import base64
import httpx
from concurrent.futures import ProcessPoolExecutor, as_completed

from decord import VideoReader
from PIL import Image
import numpy as np
import random
import re
import argparse
from datetime import datetime
import copy
from tqdm import tqdm

import base64
from io import BytesIO

from pydantic import BaseModel
from enum import Enum


class VQAscore(BaseModel):
    reasoning: str
    score: List[int]


# 生成 JSON Schema
schema = VQAscore.model_json_schema()





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_max_num', default=None, type=int)
    parser.add_argument('--api', default="http://localhost:8000/v1", type=str)
    parser.add_argument('--api_key', default="EMPTY", type=str)
    parser.add_argument('--root_path', default='/mnt/data/', type=str)
    parser.add_argument('--model_name', default='/mnt/data/', type=str)
    parser.add_argument('--data_path', default='/mnt/data/', type=str)
    parser.add_argument('--out_path', default='/mnt/data/', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--req_size', default=1, type=int)
    parser.add_argument('--out_freq', default=1, type=int)
    parser.add_argument('--num_work', default=1, type=int)
    parser.add_argument('--max_tokens', default=32768, type=int)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--num_frames', type=int, default=8, help="max image patch num")
    parser.add_argument('--start_frames', type=int, default=1, help="start frames")
    parser.add_argument('--end_frames', type=int, default=1, help="end frames")
    return parser.parse_args()

system_prompt_template = '''You are a strict evaluator. You will have to evaluate the model response reasoning chain and answer based on the reference reasoning chain and ground truth answer.
Given:
    Question: The original forecasting question with image originates from the first video frame.
    Reference Reasoning Chain: What actually happened, as a reference for the rationality of the reasoning chain.
    Ground Truth Answer: The ground truth of the question.
    Model Response Reasoning Chain: The model's reasoning chain.
    Model Response Answer: The model's answer.
The rating should base on the following rules:
    Reasoning Chain Quality: Score 0-5 based on the logical coherence, completeness, and relevance of the reasoning (including appropriate use of multimodal information if present). The chain need not match the reference exactly but must be valid and support the final answer.
    Answer Accuracy: Score 0-5 based on how well the final answer matches the ground truth answer. Full credit requires correctness and completeness; partial or incorrect answers receive lower scores.
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the
Reasoning Chain and 'score2' evaluates the Answer.
You will have to give your output in the JSON format (Keep your reasoning concise and short.):
{
"reasoning": str #the score reasoning
"score": List[int]
}
'''.strip()

def prepare_prompt(
                 question_file, #video of question
                 question, #question
                 frames, # images in question
                 recon_frames,# images in ground truth answer
                 answer, #ground truth answer
                 model_response, #model generate response with images
                 num_frames=3,
                 start_frames=0,
                 end_frames=0,
                 ):
    # assert len(question_file)==1, 'only one video per questions'
    clip = frames+recon_frames
    question_images = read_frames_decord(question_file, num_frames=num_frames, start_frames=start_frames, end_frames=end_frames, sample='middle', clip=clip)
    num_input_images = len(frames)
    num_output_images = len(recon_frames)
    question_images = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{pil_to_base64(img)}"},
    } for img in question_images]

    user_content = [{"type": "text", "text": f"Question: "}]

    image_cnt = question.count('<image>')
    image_index = 0
    #prepare question
    prompt_parts = question
    for i in range(image_cnt):
        if image_index == num_input_images:
            break
        pre_parts, prompt_parts = prompt_parts.split('<image>', 1)
        if pre_parts.strip():
            user_content.append({"type": "text", "text": f"{pre_parts}"})
        user_content.append(question_images[image_index])
        image_index=1+image_index
    if(prompt_parts.strip()):
        user_content.append({"type": "text", "text": f"{prompt_parts}"})
    #prepare ground truth answer
    reasoning = re.split(r'<ans>', answer)[0].strip()
    answer_match = re.search(r'<ans>(.*?)</ans>', answer)
    if answer_match:
        model_answer = answer_match.group(1).strip()  
    elif "</think>" in model_answer:
        tmp = model_answer.split("</think>", 1)
        answer_match = tmp[1]
        reasoning = tmp[0]
        if answer_match.strip():
            model_answer = answer_match
    else:
        model_answer = ''
    prompt_parts = reasoning
    user_content.append({"type": "text", "text": f"\nReference Reasoning Chain: "})
    for i in range(image_cnt):
        if image_index == num_input_images+num_output_images:
            break
        pre_parts, prompt_parts = prompt_parts.split('<rimage>', 1)
        if pre_parts.strip():
            user_content.append({"type": "text", "text": f"{pre_parts}"})
        user_content.append(question_images[image_index])
        image_index=1+image_index
    if(prompt_parts.strip()):
        user_content.append({"type": "text", "text": f"{prompt_parts}"})
    user_content.append({"type": "text", "text": f"\nGround Truth Answer: {answer}"})

    #prepare model_answer
    if model_response[-1].endswith('.png'):
        model_response.pop()
    model_answer = model_response[-1]
    reasoning = re.split(r'<ans>', model_answer)[0].strip()
    answer_match = re.search(r'<ans>(.*?)</ans>', model_answer)
    if answer_match:
        model_answer = answer_match.group(1).strip()  
    elif "</think>" in model_answer:
        tmp = model_answer.split("</think>", 1)
        answer_match = tmp[1]
        reasoning = tmp[0]
        if answer_match.strip():
            model_answer = answer_match
    else:
        model_answer = ''
    model_response[-1]=reasoning

    user_content.append({"type": "text", "text": f"\nModel Response Reasoning Chain: "})
    for item in model_response:
        if isinstance(item, str):
            if item.endswith('.png'):
                item = Image.open(item)
                tmp_img = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{pil_to_base64(item)}"},
                }
                user_content.append(tmp_img)
            else:
                user_content.append({"type": "text", "text": f"{item}"})
    user_content.append({"type": "text", "text": f"\nModel Response Answer: {model_answer}"})

    messages = [{"role": "system", "content": [{"type": "text", "text": f"{system_prompt_template}"}]},
                {"role": "user", "content": user_content}] 
    return messages

def pil_to_base64(image, format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

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


async def preprocess_and_queue(input_dataset, queue, batch_size, num_workers, executor):
    loop = asyncio.get_event_loop()
    
    tasks = []
    processed_count = 0
    
    for item in input_dataset:
        if processed_count >= batch_size * num_workers:
            # Wait for any task to complete before adding a new one
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                data = await fut
                if data is not None:  
                    await queue.put(data)
                tasks.remove(fut)
                processed_count -= 1
                
        # 提交任务到进程池
        future = loop.run_in_executor(
            executor,
            prepare_prompt_wrapper, 
            item
        )
        tasks.append(asyncio.ensure_future(future))
        processed_count += 1

    # 等待所有剩余的任务完成
    for fut in asyncio.as_completed(tasks):
        data = await fut
        if data is not None:
            await queue.put(data)

def prepare_prompt_wrapper(data):
    try:
        question_file = os.path.join(args.root_path, data["video"])
        ori_data = copy.deepcopy(data)
        question = data['question']
        frames = data['frames']
        recon_frames = data['recon_frames']
        answer = data['answer']
        model_response = data['model_response']
        

        messages = prepare_prompt(
            question_file=question_file,
            question=question,
            frames=frames,
            recon_frames=recon_frames,
            answer=answer,
            model_response=model_response,
            num_frames=args.num_frames,
            start_frames=args.start_frames,
            end_frames=args.end_frames,
        )
        return {'data': ori_data, 'messages': messages}
    except Exception as e:
        print(f"Error in wrapper for data {data.get('video', 'Unknown')}: {e}")
        return None


async def read_input_stream_filtered(file_path: str, skip_keys: Set[str]) -> AsyncGenerator[Dict[str, Any], None]:
    cnt = 0
    async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
        async for line in f:
            if args.sample_max_num is not None and cnt >= args.sample_max_num:
                break
            line = line.strip()
            if line:
                cnt+=1
                data = json.loads(line)
                key = data.get("video")
                if key and key in skip_keys:
                    continue
                yield data

def load_completed_keys(output_file: str) -> Set[str]:
    keys = set()
    if not os.path.exists(output_file):
        return keys

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading completed keys", unit="lines"):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    key = data.get("video")
                    if key:
                        keys.add(key)
                except Exception:
                    continue
    return keys

def load_input_data(file_path: str, skip_keys: Set[str]):
    dataset = []
    if not os.path.exists(file_path):
        return dataset
    with open(file_path, 'r', encoding='utf-8') as f:

        for line in tqdm(f, desc="Loading input keys", unit="lines"):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    key = data.get("video")
                    if key in skip_keys:
                        continue
                    dataset.append(data)
                except Exception:
                    continue
    return dataset


async def process_line(queue, result_queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        
        try:
            data = item['data']
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=item['messages'],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                extra_body={
                    'top_k':args.top_k,
                    'min_p':args.min_p,
                    # 'provider':{
                    #     'order': ['azure'],
                    #     "allow_fallbacks": False,
                    # },
                    "usage": {
                        "include": True,
                    },
                    # 'reasoning': {"effort": "minimal"},
                    'reasoning': {"effort": "none"},
                },
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "json_schema",
                        "schema": schema
                    },
                },
            )
            answer = response.choices[0].message.content
            answer = json.loads(answer)
            score = answer['score']

            data['model_chosen'] = answer['reasoning']
            data['score'] = score

        except Exception as e:
            data = {'failure':True}
            print(f"Error during chat: {e}")
        
        await result_queue.put(data)
        queue.task_done()


async def write_output_buffered(result_queue: asyncio.Queue):
    """
    从队列中消费结果，缓存写入，并在每次写入时打印时间戳和写入数量。
    """
    total_cnt = 0
    buffer = []
    while True:
        result = await result_queue.get()
        if result is None:  # 结束信号
            if buffer:
                valid_items = [item for item in buffer if not item.get('failure', False)]
                if valid_items:
                    timestamp = datetime.now().isoformat()
                    count = len(valid_items)
                    total_cnt+=count
                    async with aiofiles.open(OUTPUT_FILE, mode="a", encoding="utf-8") as f_out:
                        for item in valid_items:
                            await f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    print(f"[{timestamp}] Final write {count} to {OUTPUT_FILE}, total {total_cnt}")
            break

        buffer.append(result)
        result_queue.task_done()

        if len(buffer) >= OUT_FREQ:
            valid_items = [item for item in buffer if not item.get('failure', False)]
            if valid_items:
                timestamp = datetime.now().isoformat()
                count = len(valid_items)
                total_cnt+=count
                async with aiofiles.open(OUTPUT_FILE, mode="a", encoding="utf-8") as f_out:
                    for item in valid_items:
                        await f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"[{timestamp}] write {count} to {OUTPUT_FILE}, total {total_cnt}")
            else:
                timestamp = datetime.now().isoformat()
                print(f"[{timestamp}] got {len(buffer)} failure data, no data need wirte")

            buffer.clear()

async def main():
    print(f"start load exist data")
    completed_keys = load_completed_keys(OUTPUT_FILE)
    skip_num = len(completed_keys)
    print(f"Skipping {skip_num} already processed items")
    print(f"start load data")
    input_dataset = load_input_data(INPUT_FILE, completed_keys)
    print(f"Loading {len(input_dataset)} input dataset")
    executor = ProcessPoolExecutor(max_workers=NUM_WORKERS)

    result_queue = asyncio.Queue(maxsize=max(BATCH_SIZE * NUM_WORKERS, OUT_FREQ + BATCH_SIZE))
    writer_task = asyncio.create_task(write_output_buffered(result_queue))

    queue = asyncio.Queue(maxsize=BATCH_SIZE * NUM_WORKERS)
    
    try:
        preprocessor_task = asyncio.create_task(preprocess_and_queue(
            input_dataset, queue, BATCH_SIZE, NUM_WORKERS, executor))
        
        for _ in range(args.req_size):
            asyncio.create_task(process_line(queue, result_queue))
        
        await preprocessor_task
        
        await queue.join()
        for _ in range(args.req_size):
            await queue.put(None)
    except KeyboardInterrupt:
        print("Received Ctrl+C. Shutting down gracefully...")
        tasks = [preprocessor_task, writer_task]
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        executor.shutdown(wait=True)
        raise
    except Exception as e:
        print(f"Fatal error in main: {e}")
        raise

    finally:
        await result_queue.put(None)
        await writer_task


if __name__ == "__main__":
    args = get_args()

    INPUT_FILE = args.data_path
    OUTPUT_FILE = args.out_path
    API_key = args.api_key
    VLLM_API_BASE = args.api
    BATCH_SIZE = args.batch_size
    OUT_FREQ = args.out_freq
    MODEL_NAME = args.model_name
    NUM_WORKERS = args.num_work


    client = AsyncOpenAI(base_url=VLLM_API_BASE, api_key=API_key,     http_client=httpx.AsyncClient(
        timeout=httpx.Timeout(300, read=600)
    ))
    asyncio.run(main())