# Simple Networks for Medical Imaging (SNMI) 

A customized PyTorch framework for medical image analysis.

## Quick start
### Windows
1. Install Conda (https://www.anaconda.com/products/distribution)
2. Create Conda environment
```bash
conda create -n pytorch-env python=3.9
```
3. Install PyTorch through Conda (https://pytorch.org/get-started/locally/)
```bash
conda activate pytorch-env
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
1. Install dependencies
```bash
cd [your_path]/snmi
pip install -r requirement.txt
```

## Tensorboard

The framework use Tensorboard to log and visualize results.
While the training starts, run tensorboard on your PC:
```bash
cd [your_path]/results
tensorboard --logdir=log
```
If succeed, you will see something like:
```bash
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.9.1 at http://localhost:6006/ (Press CTRL+C to quit)
```
Then open the link shown above (http://localhost:6006/) in your browser to see the log and results.

