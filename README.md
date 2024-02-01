# InstructPix2Pix For Videos
This simple python script allows users to use the PyTorch implementation of [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix) to generate videos where each frame has been passed along the Pix2Pix Stable Diffusion pipeline with a prompt of the user's choosing.

## Requirements
This has been tested on **Python** v3.11.6 using an NVIDIA GeForce RTX 3070. A [Cuda-enabled GPU](https://developer.nvidia.com/cuda-gpus) is highly recommended.

You must also run the following command in the project directory
`pip install -r requirements.txt`


## Run
```bash
python main.py $source $fps $prompt
```

- `$source` is the path of the video that you would like to run through the InstructPix2Pix pipeline.
- `$fps` is an integer that represents the framerate of the supplied video
- `$prompt` is the string that will be used to transform the frames of the video.

## Output
**Seizure Warning**
A video `output.mp4` will be created in the project directory after the execution of the script. The output will most likely include flickering due to the nature of stable diffusion on nearly identical images.

### Examples
The GIFs are ~90MB and take some time to load.
_Source_

<img src="examples/cat.gif" width="40%" height="40%">

_"Turn the cat into a robot"_

<img src="examples/cat-robot.gif" width="40%" height="40%">


_"Turn the cat into a marble sculpture"_

<img src="examples/cat-marble.gif" width="40%" height="40%">
