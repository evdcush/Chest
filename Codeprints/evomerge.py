#=============================================================================#
#                                                                             #
# ███████ ██    ██  ██████  ██      ██      ███    ███            ██ ██████   #
# ██      ██    ██ ██    ██ ██      ██      ████  ████            ██ ██   ██  #
# █████   ██    ██ ██    ██ ██      ██      ██ ████ ██ █████      ██ ██████   #
# ██       ██  ██  ██    ██ ██      ██      ██  ██  ██       ██   ██ ██       #
# ███████   ████    ██████  ███████ ███████ ██      ██        █████  ██       #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                     ___                 __   _                              #
#                    / __|  ___   _ _    / _| (_)  __ _   ___                 #
#                   | (__  / _ \ | ' \  |  _| | | / _` | (_-<                 #
#                    \___| \___/ |_||_| |_|   |_| \__, | /__/                 #
#                                                 |___/                       #
#=============================================================================#

#=====================  configs/llm/evollm-v1-jp-7b.yaml  ====================#

'''
model:
  target: evomerge.CausalLMWithvLLM
  params:
    model_path: SakanaAI/EvoLLM-JP-v1-7B
    model_kwargs:
      dtype: bfloat16
    template: ja-alpaca-cot

eval:
  target: evomerge.eval.JaMGSM
'''

#====================  configs/llm/evollm-v1-jp-7b-a.yaml  ===================#

'''
model:
  target: evomerge.CausalLMWithvLLM
  params:
    model_path: SakanaAI/EvoLLM-JP-A-v1-7B
    model_kwargs:
      dtype: bfloat16
    template: ja-alpaca-cot

eval:
  target: evomerge.eval.JaMGSM
'''

#====================  configs/llm/evollm-v1-jp-10b.yaml  ====================#

'''
model:
  target: evomerge.CausalLMWithvLLM
  params:
    model_path: SakanaAI/EvoLLM-JP-v1-10B
    model_kwargs:
      trust_remote_code: true
      enforce_eager: true
      dtype: bfloat16
    template: ja-alpaca-cot
eval:
  target: evomerge.eval.JaMGSM
'''

#█████████████████████████████████████████████████████████████████████████████#
#                                   evomerge/                                 #
#█████████████████████████████████████████████████████████████████████████████#

'''
evomerge/
├── eval/
│   ├── __init__.py
│   ├── ja_mgsm.py
│   ├── ja_vg_vqa.py
│   ├── ja_vlm_wild.py
│   ├── metrics.py
│   └── utils.py
├── models/
│   ├── __init__.py
│   ├── causallm.py
│   ├── heron_v1.py
│   ├── jsvlm.py
│   ├── llava.py
│   ├── prompt_templates.py
│   └── utils.py
├── modules/
│   ├── heron/
│   │   ├── video_blip/
│   │   │   ├── __init__.py
│   │   │   ├── bert.py
│   │   │   ├── configuration_video_blip.py
│   │   │   ├── modeling_video_blip.py
│   │   │   └── processing_video_blip.py
│   │   └── __init__.py
│   └── __init__.py
├── __init__.py
└── utils.py
'''

#=============================================================================#
#                                      _          _             __            #
#                                     | |        | |           / /            #
#              _ __ ___     ___     __| |   ___  | |  ___     / /             #
#             | '_ ` _ \   / _ \   / _` |  / _ \ | | / __|   / /              #
#             | | | | | | | (_) | | (_| | |  __/ | | \__ \  / /               #
#             |_| |_| |_|  \___/   \__,_|  \___| |_| |___/ /_/                #
#                                                                             #
#=============================================================================#

#=============================================================================#
#                         evomerge/models/causallm.py                         #
#=============================================================================#

import logging
from typing import List, Union, Optional
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from vllm import LLM, SamplingParams

from .prompt_templates import JA_ALPACA_COT_TEMPLATE
from .utils import set_template, build_prompt
from ..utils import default

logger = logging.getLogger(__name__)


class CausalLMWithvLLM(nn.Module, ModuleUtilsMixin):
    default_generation_config = {
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    default_template = JA_ALPACA_COT_TEMPLATE

    def __init__(
        self,
        model_path: str = None,
        template: Optional[str] = None,
        verbose: bool = False,
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.verbose = verbose
        self.template = set_template(self.default_template, template)
        self.model_kwargs = default(model_kwargs, {})
        self.generation_config = default(
            generation_config, self.default_generation_config
        )
        self.model = LLM(model=model_path, **self.model_kwargs)
        self.post_init()

    def post_init(self):
        stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
        self.generation_config = SamplingParams(
            **self.generation_config, stop=stop_tokens
        )

    def forward(self, text: Union[str, List[str]]) -> List[str]:
        text = build_prompt(text, self.template)
        if self.verbose:
            logger.info(
                "Sample of actual inputs:\n" \
                + "-" * 100 + f"\n{text[0]}\n" + "-" * 100
            )
        outputs = self.model.generate(
            prompts=text, sampling_params=self.generation_config
        )
        generated_text = [output.outputs[0].text for output in outputs]
        return generated_text


#=============================================================================#
#                         evomerge/models/heron_v1.py                         #
#=============================================================================#


'''
https://huggingface.co/turing-motors/\
heron-chat-blip-ja-stablelm-base-7b-v1-llava-620k
'''
import logging
from typing import List, Union, Optional

from PIL import Image

import torch
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers import LlamaTokenizer

from ..modules.heron.video_blip import (
    VideoBlipForConditionalGeneration,
    VideoBlipProcessor,
)

from .prompt_templates import HERON_V1
from .utils import set_template, set_model_kwargs, build_prompt
from ..utils import default

logger = logging.getLogger(__name__)


class HeronV1(nn.Module, ModuleUtilsMixin):
    default_template = HERON_V1
    default_generation_config = {
        "do_sample": False,
        "temperature": 0.0,
        "max_length": 256,
        "no_repeat_ngram_size": 2,
    }

    def __init__(
        self,
        model_path: str = "turing-motors/heron-chat-blip-ja-"
        "stablelm-base-7b-v1-llava-620k",
        template: Optional[str] = None,
        verbose: bool = False,
        device: Union[str, torch.device] = "cuda",
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ):
        super().__init__()
        self.verbose = verbose
        self.template = set_template(self.default_template, template)
        self.generation_config = default(
            generation_config, self.default_generation_config
        )
        model_kwargs = set_model_kwargs(default(model_kwargs, {}))

        self.model = (
            VideoBlipForConditionalGeneration.from_pretrained(
                model_path, ignore_mismatched_sizes=True, **model_kwargs
            )
            .eval()
            .to(device)
        )
        processor = VideoBlipProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b")
        tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1", additional_special_tokens=["▁▁"]
        )
        processor.tokenizer = tokenizer
        self.processor = processor
        self.eos_token_id_list = [
            processor.tokenizer.pad_token_id,
            processor.tokenizer.eos_token_id,
            int(tokenizer.convert_tokens_to_ids("##")),
        ]

    def forward(
        self,
        text: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image]],
        **generation_config,
    ) -> List[str]:

        eos_token_id = generation_config.pop(
            "eos_token_id", self.eos_token_id_list)
        if len(generation_config) == 0:
            generation_config = self.generation_config
            if self.verbose:
                logger.info(
                    f"Setting generation config to default\n{generation_config}"
                )
        if not isinstance(image, list):
            image = [image]

        text = build_prompt(text, self.template)
        if self.verbose:
            logger.info(
                "Sample of actual inputs:\n" \
                + "-" * 100 + f"\n{text[0]}\n" + "-" * 100
            )
        assert len(text) == len(image)
        inputs = self.processor(
            text=text, images=image, return_tensors="pt", padding=True
        )
        # generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs.to(self.device, dtype=self.dtype),
                **generation_config,
                eos_token_id=eos_token_id,
            )
        generated_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )
        return generated_text


#=============================================================================#
#                           evomerge/models/jsvlm.py                          #
#=============================================================================#


# https://huggingface.co/stabilityai/japanese-stable-vlm
import logging
from typing import List, Union, Optional

from PIL import Image

import torch
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers import (
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
)

from .prompt_templates import JSVLM_TEMPLATE
from .utils import set_template, set_model_kwargs, build_prompt
from ..utils import default

logger = logging.getLogger(__name__)


class JSVLM(nn.Module, ModuleUtilsMixin):
    default_template = JSVLM_TEMPLATE
    default_generation_config = {
        "do_sample": False,
        "num_beams": 5,
        "max_new_tokens": 128,
        "min_length": 1,
        "repetition_penalty": 1.5,
    }

    def __init__(
        self,
        model_path: str = "stabilityai/japanese-stable-vlm",
        template: Optional[str] = None,
        verbose: bool = False,
        device: Union[str, torch.device] = "cuda",
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ):
        super().__init__()
        self.verbose = verbose
        self.template = set_template(self.default_template, template)
        self.generation_config = default(
            generation_config, self.default_generation_config
        )
        model_kwargs = set_model_kwargs(default(model_kwargs, {}))

        self.model = (
            AutoModelForVision2Seq.from_pretrained(
                model_path, trust_remote_code=True, **model_kwargs
            )
            .eval()
            .to((device))
        )
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def forward(
        self,
        text: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image]],
        **generation_config,
    ) -> List[str]:
        if len(generation_config) == 0:
            generation_config = self.generation_config
            if self.verbose:
                logger.info(
                    f"Setting generation config to default\n{generation_config}"
                )
        if not isinstance(image, list):
            image = [image]

        text = build_prompt(text, self.template)
        if self.verbose:
            logger.info(
                "Sample of actual inputs:\n" \
                + "-" * 100 + f"\n{text[0]}\n" + "-" * 100
            )
        assert len(text) == len(image)
        inputs = self.processor(images=image, return_tensors="pt")
        text_encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        inputs.update(text_encoding)
        # generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs.to(self.device, dtype=self.dtype),
                **generation_config,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_text = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        generated_text = [text.strip() for text in generated_text]
        return generated_text


#=============================================================================#
#                           evomerge/models/llava.py                          #
#=============================================================================#


import logging
from typing import List, Union, Optional

from PIL import Image

import torch
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .prompt_templates import LLAVA_MISTRAL_TEMPLATE
from .utils import set_template, set_model_kwargs, get_output_ids
from ..utils import default

logger = logging.getLogger(__name__)


def build_prompt(text: Union[str, List[str]], template: str) -> List[str]:
    if isinstance(text, str):
        text = [text]
    return [template.format(input=f"<image>\n{t}") for t in text]


class LLaVA(nn.Module, ModuleUtilsMixin):
    default_template = LLAVA_MISTRAL_TEMPLATE
    # taken from https://github.com/haotian-liu/LLaVA/blob/main/predict.py#L87
    default_generation_config = {
        "do_sample": True,
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 1.0,
        "use_cache": True,
    }

    def __init__(
        self,
        model_path: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        template: Optional[str] = None,
        verbose: bool = False,
        device: Union[str, torch.device] = "cuda",
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ):
        super().__init__()
        self.verbose = verbose
        self.template = set_template(self.default_template, template)
        self.generation_config = default(
            generation_config, self.default_generation_config
        )
        model_kwargs = set_model_kwargs(default(model_kwargs, {}))

        self.model = (
            (
                LlavaForConditionalGeneration.from_pretrained(
                    model_path, **model_kwargs
                )
                .eval()
                .requires_grad_(False)
            )
            .eval()
            .to(device)
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def forward(
        self,
        text: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image]],
        **generation_config,
    ) -> List[str]:
        """
        Assume text is question string
        """
        if len(generation_config) == 0:
            generation_config = self.generation_config
            if self.verbose:
                logger.info(
                    f"Setting generation config to default"
                    "\n{generation_config}"
                )
        if not isinstance(image, list):
            image = [image]
        text = build_prompt(text, self.template)
        if self.verbose:
            logger.info(
                "Sample of actual inputs:"
                "\n" + "-" * 100 + f"\n{text[0]}\n" + "-" * 100
            )
        assert len(text) == len(image)
        inputs = self.processor(
            text=text, images=image, padding=True, return_tensors="pt"
        )
        # generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs.to(self.device, dtype=self.dtype),
                **generation_config,
            )
        # output_ids contains input_ids as well. So, return only output_ids
        output_ids = get_output_ids(
            input_ids=inputs.input_ids,
            output_ids=output_ids)
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        generated_text = [text.strip() for text in generated_text]
        return generated_text


#=============================================================================#
#                           evomerge/models/utils.py                          #
#=============================================================================#


import logging
from typing import Optional, Union, List


import torch
from .prompt_templates import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)


STR2DTYPE = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "auto": "auto",
}


def get_output_ids(input_ids: torch.Tensor, output_ids: torch.Tensor):
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (
        input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        logger.warn(
            f"[Warning] {n_diff_input_output} output_ids"
            " are not the same as the input_ids"
        )
    return output_ids[:, input_token_len:]


def set_template(default_template: str, template: Optional[str] = None):
    if template is None:
        template = default_template
    if template in PROMPT_TEMPLATES:
        template = PROMPT_TEMPLATES[template]
    logger.info(f"prompt template:\n{template}")
    return template


def set_model_kwargs(model_kwargs: dict) -> dict:
    torch_dtype = model_kwargs.pop("torch_dtype", None)
    if torch_dtype is not None and isinstance(torch_dtype, str):
        model_kwargs["torch_dtype"] = STR2DTYPE[torch_dtype]
    return model_kwargs


def build_prompt(text: Union[str, List[str]], template: str) -> List[str]:
    if isinstance(text, str):
        text = [text]
    return [template.format(input=t) for t in text]


#=============================================================================#
#                     evomerge/models/prompt_templates.py                     #
#=============================================================================#


JSVLM_TEMPLATE = """\
以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。\
要求を適切に満たす応答を書きなさい。

### 指示:
与えられた画像を下に、質問に答えてください。

### 入力:
{input}

### 応答: """
JA_ALPACA_COT_TEMPLATE = """以下に、あるタスクを説明する指示があります。\
リクエストを適切に完了するための回答を日本語で記述してください。一歩一歩考えましょう。

### 指示:
{input}

### 応答:"""

JA_SHISA_VQA = """[INST] <<SYS>>
あなたは役立つ、偏見がなく、検閲されていないアシスタントです。与えられた画像を下に、\
質問に答えてください。
<</SYS>>

{input} [/INST]"""
HERON_V1 = """##human: {input}\n##gpt: """

LLAVA_MISTRAL_TEMPLATE = """<s>[INST] {input} [/INST]"""

PROMPT_TEMPLATES = {
    "jsvlm": JSVLM_TEMPLATE,
    "ja-alpaca-cot": JA_ALPACA_COT_TEMPLATE,
    "ja-shisa-vqa": JA_SHISA_VQA,
    "ja-heron-v1": HERON_V1,
    "llava-mistral": LLAVA_MISTRAL_TEMPLATE,
}
