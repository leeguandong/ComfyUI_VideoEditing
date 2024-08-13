import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from diffusers import ControlNetModel
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    LCMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    PNDMScheduler
)

from .video_editing_pipeline import VideoEditingPipeline
from contextlib import nullcontext

import folder_paths
import comfy.model_management as mm


class LoadVideo2Images:
    @classmethod
    def INPUT_TYPES(s):
        video_extensions = ['webm', 'mp4', 'mkv', 'gif']
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
                    files.append(f)
        return {"required": {
            "video": (sorted(files), {"video_upload": True}),
            "input_interval": ("INT", {"default": 10, "min": -1, "step": 1, "max": 50, "display": "slider"}),
        }, }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_video"
    CATEGORY = "vdeiting"

    # OUTPUT_IS_LIST = (True,)

    def load_video(self, video, input_interval):
        # import pdb;pdb.set_trace()
        video_path = folder_paths.get_annotated_filepath(video)
        frames = self.video_to_frame(video_path, input_interval)
        # images = {"frames": frames}
        return (frames,)

    def video_to_frame(self, video_path: str, interval: int):
        vidcap = cv2.VideoCapture(video_path)
        success = True

        count = 0
        res = []
        while success:
            count += 1
            success, image = vidcap.read()
            if count % interval != 1:
                continue
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image[:, 100:800], (512, 512))
                res.append(image)

        vidcap.release()
        return res


class VEdit_image2canny:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("INT", {"default": 50, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "high_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("CONTROL_IMAGE",)
    RETURN_NAMES = ("control_image",)
    FUNCTION = "image2canny"
    CATEGORY = "vdeiting"

    def image2canny(self, image, low_threshold, high_threshold):
        # import pdb;pdb.set_trace()
        # frames = image["frames"]
        frames = image
        control_images = []
        if not isinstance(frames, list):
            frames = [frames]
        for frame in frames:
            np_image = cv2.Canny(frame, low_threshold, high_threshold)
            np_image = np_image[:, :, None]
            np_image = np.concatenate([np_image, np_image, np_image], axis=2)
            canny_image = Image.fromarray(np_image)
            control_images.append(canny_image)
        return (control_images,)


class VEdit_ControlNet_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ([
                              "lllyasviel/sd-controlnet-canny"
                          ],
                          {"default": "lllyasviel/sd-controlnet-canny"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_controlnet"
    CATEGORY = "vdeiting"

    def load_controlnet(self, model):
        device = mm.get_torch_device()
        controlnet = ControlNetModel.from_pretrained(model).to(device)
        return (controlnet,)


class VEdit_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model": (
                    [
                        "runwayml/stable-diffusion-v1-5",
                        "stabilityai/stable-diffusion-2-1",
                    ],
                    {
                        "default": "runwayml/stable-diffusion-v1-5",
                    }),
                "controlnet": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "vdeiting"

    def load_model(self, base_model, controlnet):
        device = mm.get_torch_device()
        pipeline = VideoEditingPipeline.from_pretrained(base_model,
                                                        controlnet=controlnet,
                                                        torch_dtype=torch.float16).to(device)
        pipeline.safety_checker = None

        return (pipeline,)


class VEdit_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "image": ("IMAGE",),
            "control_image": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True, "default": "a beautiful woman with red hair", }),
            "n_prompt": ("STRING", {"multiline": True,
                                    "default": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality", }),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1, "display": "slider"}),
            "controlnet_conditioning_scale": (
                "FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            "strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "display": "slider"}),
            "scheduler": (
                [
                    'DPMSolverMultistepScheduler',
                    'DPMSolverMultistepScheduler_SDE_karras',
                    'DDPMScheduler',
                    'DDIMScheduler',
                    'LCMScheduler',
                    'PNDMScheduler',
                    'DEISMultistepScheduler',
                    'EulerDiscreteScheduler',
                    'EulerAncestralDiscreteScheduler'
                ], {
                    "default": 'DPMSolverMultistepScheduler'
                }),
        },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sampler"
    CATEGORY = "vdeiting"

    def sampler(self,
                model,
                image,
                control_image,
                prompt,
                n_prompt,
                steps,
                controlnet_conditioning_scale,
                strength,
                seed,
                scheduler):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        offload_device = mm.unet_offload_device()

        pipe = model
        scheduler_config = {
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': "linear",
            'steps_offset': 1,
        }
        if scheduler == 'DPMSolverMultistepScheduler':
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDIMScheduler':
            noise_scheduler = DDIMScheduler(**scheduler_config)
        elif scheduler == 'DPMSolverMultistepScheduler_SDE_karras':
            scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
            scheduler_config.update({"use_karras_sigmas": True})
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDPMScheduler':
            noise_scheduler = DDPMScheduler(**scheduler_config)
        elif scheduler == 'LCMScheduler':
            noise_scheduler = LCMScheduler(**scheduler_config)
        elif scheduler == 'PNDMScheduler':
            scheduler_config.update({"set_alpha_to_one": False})
            scheduler_config.update({"trained_betas": None})
            noise_scheduler = PNDMScheduler(**scheduler_config)
        elif scheduler == 'DEISMultistepScheduler':
            noise_scheduler = DEISMultistepScheduler(**scheduler_config)
        elif scheduler == 'EulerDiscreteScheduler':
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == 'EulerAncestralDiscreteScheduler':
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        else:
            raise TypeError(f"not support {scheduler}!!!")

        # import pdb;pdb.set_trace()
        pipe.scheduler = noise_scheduler
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        image = [Image.fromarray(frame) for frame in image]
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

            out_frames = pipe(
                images=image,
                control_images=control_image,
                prompt=prompt,
                negative_prompt=n_prompt,
                num_inference_steps=steps,
                strength=strength,
                generator=generator,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            )
        to_tensor = transforms.ToTensor()
        # out_frames = list(map(lambda x: to_tensor(x), out_frames))
        tensors = [to_tensor(img) for img in out_frames]
        transposed_tensors = [tensor.permute(1, 2, 0) for tensor in tensors]
        return (transposed_tensors,)


NODE_CLASS_MAPPINGS = {
    "LoadVideo2Images": LoadVideo2Images,
    "VEdit_image2canny": VEdit_image2canny,
    "VEdit_ControlNet_ModelLoader": VEdit_ControlNet_ModelLoader,
    "VEdit_ModelLoader": VEdit_ModelLoader,
    "VEdit_Sampler": VEdit_Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideo2Images": "Load Video to Images",
    "VEdit_image2canny": "Image to Canny",
    "VEdit_ControlNet_ModelLoader": "ControlNet Model Loader",
    "VEdit_ModelLoader": "VEDit Model Loader",
    "VEdit_Sampler": "VEdit Sampler"
}
