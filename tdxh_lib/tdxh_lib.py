# _*_ coding:utf-8 _*_
import os
import sys
from PIL import Image
# import numpy as np

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
fooocus_dir = os.path.abspath(os.path.join(my_dir,'..'))
sys.path.append(my_dir)
# print(f"{my_dir}\n{fooocus_dir}")

SD_XL_BASE_RATIOS = {
    "0.25":(512, 2048), # new
    "0.26":(512, 1984), # new
    "0.27":(512, 1920), # new
    "0.28":(512, 1856), # new
    "0.32":(576, 1792), # new
    "0.33": (576, 1728), # new
    "0.35": (576, 1664), # new
    "0.4":(640, 1600), # new
    "0.42": (640, 1536), # new
    "0.48":(704, 1472), # new
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704), # not in training value but in Stability-AI/generative-models/scripts/demo/sampling.py
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
    "3.11":(1792, 576), # new
    "3.62":(1856, 512), # new
    "3.75":(1920, 512), # new
    "3.88":(1984, 512), # new
    "4.0":(2048, 512), # new
    }

target_sizes_show = [f"{k}:{v}" for k, v in SD_XL_BASE_RATIOS.items()]


def get_SDXL_best_size(image_size = None, ratio = None):
    """
    input a tuple such as get_SDXL_best_size((1200, 900)
    or input a float num such as get_SDXL_best_size(ratio=1.3)
    return a tuple for SDXL such as (1152, 832)
    """
    best_size = None
    if image_size:
        if image_size[0] > 0 and image_size[1] > 0:
            ratio = image_size[0] / image_size[1] # w, h = image_size
    if ratio:
        target_sizes = [v for _, v in SD_XL_BASE_RATIOS.items()]
        min_diff = float('inf') # a variable to store the minimum difference
        for target_size in target_sizes:
            target_ratio = target_size[0] / target_size[1]
            diff = abs(ratio - target_ratio)
            if diff < min_diff:
                min_diff = diff
                best_size = target_size
    return best_size

def np2pil(numpy_array):
    return Image.fromarray(numpy_array) 

def tdxh_image_to_size(image):
    image = np2pil(image)
    if image.size:
        w, h = image.size[0], image.size[1]
    else:
        w, h = 0, 0
    return (w,h)

def tdxh_image_to_SDXL_best_size(image):
    return get_SDXL_best_size(tdxh_image_to_size(image))

input_language_list= [
        r"中文", 
        r"عربية", 
        r"Deutsch", 
        r"Español", 
        r"Français", 
        r"हिन्दी", 
        r"Italiano", 
        r"日本語", 
        r"한국어", 
        r"Português", 
        r"Русский", 
        r"Afrikaans", 
        r"বাংলা", 
        r"Bosanski", 
        r"Català", 
        r"Čeština", 
        r"Dansk", 
        r"Ελληνικά", 
        r"Eesti", 
        r"فارسی", 
        r"Suomi", 
        r"ગુજરાતી", 
        r"עברית", 
        r"हिन्दी", 
        r"Hrvatski", 
        r"Magyar", 
        r"Bahasa Indonesia", 
        r"Íslenska", 
        r"Javanese", 
        r"ქართული", 
        r"Қазақ", 
        r"ខ្មែរ", 
        r"ಕನ್ನಡ", 
        r"한국어", 
        r"ລາວ", 
        r"Lietuvių", 
        r"Latviešu", 
        r"Македонски", 
        r"മലയാളം", 
        r"मराठी", 
        r"Bahasa Melayu", 
        r"नेपाली", 
        r"Nederlands", 
        r"Norsk", 
        r"Polski",
        r"Română", 
        r"සිංහල", 
        r"Slovenčina", 
        r"Slovenščina", 
        r"Shqip",  
        r"Turkish", 
        r"Tiếng Việt",
    ]
# python -m pip install sentencepiece
# python -m pip install protobuf==3.20.0
input_language_default=input_language_list[0]
from tdxh_translator import Prompt,TranslatorScript
class TdxhStringInputTranslator:
    def __init__(self, input_language = input_language_default,string_value="", bool_int=1) -> None:
        self.input_language = input_language
        self.string_value=string_value
        self.bool_int=bool_int
        prompt_list=[str(self.string_value)]
        self.p_in = Prompt(prompt_list, [""])
        self.translator = TranslatorScript()
    def prompt_input(self,string_input):
        prompt_list=[str(string_input)]
        self.p_in = Prompt(prompt_list, [""])
    def active(self):
        self.translator.set_active()
    def deactive(self):
        self.translator.set_deactive()
    def run(self):
        if self.bool_int == 0:
            return (self.string_value,)
        if not hasattr(self.translator, "translator"):
            self.translator.set_active()
        self.translator.process(self.p_in,self.input_language)
        string_value_out=self.p_in.positive_prompt_list[0] \
            if self.p_in.positive_prompt_list is not None else ""
        return (string_value_out,)

def test():
    p = TdxhStringInputTranslator(string_value="建筑渲染图，现代建筑，超高层建筑，白天")
    print(p.tdxh_value_output())

# test()

