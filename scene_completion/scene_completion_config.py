from omegaconf import OmegaConf

class SceneCompletionConfig:
    def __init__(self) -> None:
        self.config_dict = {
            "device": "cuda",
            "prompt": 'a realistic photo of an empty room',
            "negative_prompt": "blurry, bad art, blurred, text, watermark, plant, nature, people, person, face, human, animal",
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
        }

    def get_config(self):
        return OmegaConf.create(self.config_dict)

