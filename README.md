# WALC-IRSR: Efficient Window Attention Meets Large-Kernel Convolution for Infrared Image Super-Resolution

WALC-IRSR is a lightweight infrared image super-resolution model that decouples long-range structure modeling and local detail refinement. Each WALC block applies large-window attention for global context, then uses an Attention–Convolution Fusion (ACF) module with shared large-kernel convolutions and local window attention to stabilize structure under noise while preserving textures. FlashAttention is used for attention efficiency, and geometric reparameterization is applied to the shared large kernel for better structural consistency.




## Installation

```bash
cd WALC
conda create -n walc python=3.10
conda activate walc
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
python setup.py develop
```

## Training

### Single GPU
```bash
python walc/train.py -opt options/train/WALC_hw_X2.yaml
```



## Testing
```bash
python walc/test.py -opt options/test/WALC_hw_X2.yaml
```



## Acknowledgement
This work is based on BasicSR and related restoration toolkits. We thank the original authors and the open-source community.
 
