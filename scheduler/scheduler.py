import json
from pathlib import Path  

from diffusers import DDIMScheduler
from diffusers import DDPMScheduler
from diffusers import PNDMScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import KDPM2AncestralDiscreteScheduler


# 获取当前文件的目录路径  
global_current_dir = Path(__file__).resolve().parent  
global_scheduler_path = global_current_dir / 'json'


class SchedulerParameters:  
    @classmethod  
    def INPUT_TYPES(cls):  
        with open(global_scheduler_path / "scheduler.json", 'r') as f:  
            scheduler_data = json.load(f)  
        cls.scheduler_list = scheduler_data["scheduler"]  
        return {  
            "required": {  
                "scheduler": (cls.scheduler_list,),  # 输入调度器
                "use_default": ("BOOLEAN", {"default": True}),
                "beta_start": ("FLOAT", {"default": 0.00085, "min": 0.0, "max": 1.0, "step": 0.0001}),  
                "beta_end": ("FLOAT", {"default": 0.012, "min": 0.0, "max": 1.0, "step": 0.0001}),  
                "num_train_timesteps": ("INT", {"default": 1000, "min": 1, "max": 5000, "step": 1}),  
                "beta_schedule": (["scaled_linear", "linear", "cosine"],),  
                "set_alpha_to_one": ("BOOLEAN", {"default": False}),  
                "skip_prk_steps": ("BOOLEAN", {"default": True}),  
                "clip_sample": ("BOOLEAN", {"default": False}),  
            },  
        }  
    
    RETURN_TYPES = ("SCHEDULER",)  
    RETURN_NAMES = ("scheduler",)  
    FUNCTION = "modify_scheduler_parameters"

    CATEGORY = "SchedulerModification"

    def modify_scheduler_parameters(self, scheduler, use_default,
                                    beta_start, beta_end, 
                                    num_train_timesteps, 
                                    beta_schedule, 
                                    set_alpha_to_one, 
                                    skip_prk_steps, 
                                    clip_sample):  
        # 将字符串转换为类并实例化  
        scheduler_class = globals()[scheduler]  # 使用 globals() 获取类  
        scheduler_instance = scheduler_class()  # 实例化类  

        scheduler_instance.from_pretrained(global_scheduler_path, subfolder=scheduler)

        if not use_default :
            scheduler_instance.beta_start = beta_start  
            scheduler_instance.beta_end = beta_end  
            scheduler_instance.num_train_timesteps = num_train_timesteps  
            scheduler_instance.beta_schedule = beta_schedule  
            scheduler_instance.set_alpha_to_one = set_alpha_to_one  
            scheduler_instance.skip_prk_steps = skip_prk_steps  
            scheduler_instance.clip_sample = clip_sample  
        
        return (scheduler_instance,)  

NODE_CLASS_MAPPINGS = {
    "SchedulerParameters": SchedulerParameters,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SchedulerParameters": "Smell Scheduler Parameters",
}