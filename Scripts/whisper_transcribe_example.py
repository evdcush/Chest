# coding: utf-8
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
get_ipython().run_line_magic('pinfo', 'AutoModelForSpeechSeq2Seq.from_pretrained')
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('ls', '')
model_path = '/home/evan/Models/Talk/whisper-large-v3'
device = "cuda:1" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device
model_id = model_path
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
path_to_vid = "/home/evan/Media/Videos/Edu/Karpathy/Andrej Karpathy - Let's build GPT： from scratch, in code, spelled out.mkv"
path_to_opus = "/home/evan/Media/Videos/Edu/Karpathy/karpathy_build_gpt_scratch.opus"
path_to_mp3 = "/home/evan/Media/Videos/Edu/Karpathy/karpathy_build_gpt_scratch.mp3"
result = pipe(path_to_vid, generate_kwargs={'language': 'english'}, return_timestamps=True)
result.keys()
result['chunks']
len(result['chunks'])
type(result)
result['text']
type(result['text'])

from datetime import datetime, timedelta
import pysrt
import io

def convert_time(data):
    seconds, milliseconds = map(int, str(data).split('.'))
    time_delta = timedelta(seconds=seconds, milliseconds=milliseconds)
    base_time = datetime(2000, 1, 1)

    result_time = base_time + time_delta
    result_str = result_time.strftime("%H:%M:%S.%f")[:-3]

    return result_str


def hf_pipeline_to_srt(json_result, output_file=None):
    file = pysrt.SubRipFile()
    for idx, chk in enumerate(json_result["chunks"]):
        text = chk["text"]
        start, end = map(convert_time, chk["timestamp"])
        
        sub = pysrt.SubRipItem(idx, 
            start=start, end=end, text=text.strip())
        file.append(sub)
        
    if output_file is not None:
        print(f"Saved to {output_file}")
        file.save(output_file)
        return output_file
    else:
        import io
        fp = io.StringIO("")
        file.write_into(fp)
        json_result = fp.getvalue()
        return json_result
        
output_file = 'karpathy_gpt_scratch.srt'
outfile = hf_pipeline_to_srt(result, output_file=output_file)
