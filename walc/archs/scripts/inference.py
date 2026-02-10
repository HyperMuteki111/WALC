import argparse
import cv2
import glob
import numpy as np
import os
import torch

from walc.archs.walc_arch import WALC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='WALC',
        choices=['WALC', 'WALCLight', 'WALCXL', 'WALCFP', 'WALCReal', 'WALCRealM'],
        help='model name, choose from [WALC, WALCLight, WALCXL, WALCFP, WALCReal, WALCRealM]'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=4,
        choices=[2, 3, 4],
        help='super resolution scale, choose from [2, 3, 4]; WALCReal only supports 4x scale'
    )
    parser.add_argument(
        '--attn_type',
        default='Flex',
        type=str,
        choices=['Naive', 'SDPA', 'Flex'],
        help='attention funcions. For detailed information, please refer to walc/archs/walc_arch.py'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='pretrained_models/WALC_DIV2K_X2.pth',
        help='path to the pre-trained model file'
    )
    parser.add_argument('--input', type=str, default='datasets/Set14/LRbicx4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/WALC', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model_class = {
        'WALC': WALC,
        'WALCLight': WALC,
        'WALCXL': WALC,
        'WALCFP': WALC,
        'WALCReal': WALC,
        'WALCRealM': WALC
    }[args.model]
    model_args = {
        'WALC': {
            'dim': 64, 'pdim': 16, 'kernel_size': 13, 'n_blocks': 5, 'conv_blocks': 5,
            'window_size': 32, 'num_heads': 4, 'exp_ratio': 1.25, 
            'upscaling_factor': args.scale, 'attn_type': args.attn_type
        },
        'WALCLight': {
            'dim': 64, 'pdim': 16, 'kernel_size': 13, 'n_blocks': 3, 'conv_blocks': 5,
            'window_size': 32, 'num_heads': 4, 'exp_ratio': 1.25, 
            'upscaling_factor': args.scale, 'attn_type': args.attn_type
        },
        'WALCXL': {
            'dim': 192, 'pdim': 48, 'kernel_size': 13, 'n_blocks': 8, 'conv_blocks': 5,
            'window_size': 48, 'num_heads': 12, 'exp_ratio': 1.25, 'use_ln': True,
            'upscaling_factor': args.scale, 'attn_type': args.attn_type
        },
        'WALCFP': {
            'dim': 48, 'pdim': 16, 'kernel_size': 13, 'n_blocks': 5, 'conv_blocks': 5,
            'window_size': 32, 'num_heads': 3, 'exp_ratio': 1.25, 
            'upscaling_factor': args.scale, 'attn_type': args.attn_type
        },
        'WALCReal': {
            'dim': 64, 'pdim': 16, 'kernel_size': 13, 'n_blocks': 10, 'conv_blocks': 5,
            'window_size': 32, 'num_heads': 4, 'exp_ratio': 2,
            'upscaling_factor': 4, 'attn_type': args.attn_type
        },
        'WALCRealM': {
            'dim': 64, 'pdim': 16, 'kernel_size': 13, 'n_blocks': 10, 'conv_blocks': 5,
            'window_size': 32, 'num_heads': 4, 'exp_ratio': 1.25,
            'upscaling_factor': args.scale, 'attn_type': args.attn_type
        }
    }[args.model]
    model = model_class(**model_args)
    model.load_state_dict(torch.load(args.model_path)['params_ema'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_{args.model}.png'), output)


if __name__ == '__main__':
    main()
    
