from omegaconf import OmegaConf
import os 

class SceneCompletionConfig:
    def __init__(self) -> None:
        self.config_dict = {
            "device": "cuda",
            "prompt": 'a realistic photo of an empty room',
            "negative_prompt": "blurry, bad art, blurred, text, watermark, plant, nature, people, person, face, human, animal",
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "config_file": "./mask2former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
            "predictions": True,
            "opts": ['MODEL.WEIGHTS', os.path.join(os.getcwd(),"mask2former/model_weights/model_final_47429163_0.pkl")],
            "confidence_threshold": 0.5,
            "n": 1,
            "p": 0
        }

    def get_config(self):
        return OmegaConf.create(self.config_dict)

