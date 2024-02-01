from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import torch
import ffmpeg
import sys
import os
import shutil

def diffuse(imgname, prompt, pipe):
    # turn image into usable pillow image object
    image = Image.open(f'input/{imgname}')
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # feed image through pipeline
    images = pipe(prompt, image=image, num_inference_steps=15, image_guidance_scale=1).images

    # save resulting image to output file
    images[0].save(f'output/{imgname}')    

def main():
    source = sys.argv[1]
    fps = sys.argv[2]
    prompt = sys.argv[3]

    # create directories for storing images
    os.mkdir('input')
    os.mkdir('output')

    # turn video to images
    ffmpeg.input(source).output('input/frame-%d.png', vf=f'fps={fps}').run(quiet=True)

    # initialize diffusion pipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # run stable difussion on the frames of the source video
    for imgname in os.listdir('input/'):
        diffuse(imgname, prompt, pipe)

    # turn diffused images into a video
    ffmpeg.input('output/frame-%d.png', pattern_type='sequence', framerate=fps).output('output.mp4', pix_fmt='yuv420p').run()

    # delete input and output directories
    shutil.rmtree('input')
    shutil.rmtree('output')


if __name__ == '__main__':
    main()
