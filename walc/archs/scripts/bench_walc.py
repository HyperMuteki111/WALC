import os
import argparse
import time
import threading
import numpy as np
import cv2
import torch
import tqdm
from contextlib import nullcontext
from torch.backends import cudnn

from walc.archs.walc_arch import WALC
from fvcore.nn import FlopCountAnalysis


class PowerMeasurer:
    def __init__(self, tick_interval=1):
        self.tick_interval = tick_interval
        self.power_usage = []
        self.stop_event = threading.Event()
        self.power_thread = None
    def start(self):
        def get_power_usage():
            time.sleep(1)
            while not self.stop_event.is_set():
                self.power_usage.append(torch.cuda.power_draw(0))
                time.sleep(self.tick_interval)
        self.stop_event.clear()
        self.power_thread = threading.Thread(target=get_power_usage)
        self.power_thread.start()
    def stop(self):
        self.stop_event.set()
        if self.power_thread is not None:
            self.power_thread.join()
            self.power_thread = None
    def average(self):
        if self.power_usage:
            return float(np.mean(self.power_usage) / 1000)
        return None


def build_model(args):
    model_kwargs = {
        'dim': args.dim,
        'pdim': args.pdim,
        'kernel_size': args.kernel_size,
        'n_blocks': args.n_blocks,
        'conv_blocks': args.conv_blocks,
        'window_size': args.window_size,
        'num_heads': args.num_heads,
        'upscaling_factor': args.scale,
        'exp_ratio': args.exp_ratio,
        'attn_type': args.attn,
        'use_ln': args.use_ln,
    }
    return WALC(**model_kwargs)


def load_image(image_path, scale):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    lr = lr.astype(np.float32) / 255.0
    t = torch.from_numpy(np.transpose(lr, (2, 0, 1))).unsqueeze(0)
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--attn', type=str, default='Flex')
    parser.add_argument('--attn_flops', type=str, default='Naive')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--pdim', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=13)
    parser.add_argument('--n_blocks', type=int, default=5)
    parser.add_argument('--conv_blocks', type=int, default=5)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--exp_ratio', type=float, default=1.25)
    parser.add_argument('--use_ln', action='store_true')
    args = parser.parse_args()

    cudnn.benchmark = True
    if args.image is not None:
        x = load_image(args.image, args.scale)
    else:
        shape = (1, 3, args.height // args.scale, args.width // args.scale)
        x = torch.FloatTensor(*shape).uniform_(0.0, 1.0)

    model = build_model(args)
    x = x.cuda()
    model = model.cuda()
    model.eval()
    if args.jit:
        with torch.no_grad():
            model = torch.jit.trace(model, x)

    context = torch.cuda.amp.autocast if args.fp16 else nullcontext
    measure_power = PowerMeasurer()
    with context():
        with torch.inference_mode():
            for _ in tqdm.tqdm(range(args.warmup)):
                model(x)
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            measure_power.start()
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timings = np.zeros((args.repeat, 1))
            for rep in tqdm.tqdm(range(args.repeat)):
                starter.record()
                model(x)
                ender.record()
                torch.cuda.synchronize()
                timings[rep] = starter.elapsed_time(ender)
            measure_power.stop()

    avg = float(np.sum(timings) / args.repeat)
    med = float(np.median(timings))
    mem_alloc = float(torch.cuda.max_memory_allocated() / 1024**2)
    mem_reserved = float(torch.cuda.max_memory_reserved() / 1024**2)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('CUDNN Benchmark:', cudnn.benchmark)
    print('Input shape:', tuple(x.shape))
    print('Average time: %.5f ms' % avg)
    print('Median time: %.5f ms' % med)
    print('Maximum GPU memory Occupancy: %.5f MB' % mem_alloc)
    print('Maximum GPU memory Reserved: %.5f MB' % mem_reserved)
    print('Params: %.3fK' % (params / 1000))
    print('Average power usage:', measure_power.average(), 'W')

    with torch.no_grad():
        flops_model_kwargs = {
            'dim': args.dim,
            'pdim': args.pdim,
            'kernel_size': args.kernel_size,
            'n_blocks': args.n_blocks,
            'conv_blocks': args.conv_blocks,
            'window_size': args.window_size,
            'num_heads': args.num_heads,
            'upscaling_factor': args.scale,
            'exp_ratio': args.exp_ratio,
            'attn_type': args.attn_flops,
            'use_ln': args.use_ln,
        }
        model_flops = WALC(**flops_model_kwargs).cuda().eval()
        flops = FlopCountAnalysis(model_flops, x).total()
        print('FLOPs: %.2f G' % (flops / 1e9))


if __name__ == '__main__':
    main()
