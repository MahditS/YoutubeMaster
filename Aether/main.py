import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pytubefix import YouTube
import pydub

yt = YouTube(input("Enter the URL of the video you want to summarize: \n>> "))

audio_stream = yt.streams.filter(only_audio=True).first()
file = audio_stream.download(output_path="audios")

ma_audio = pydub.AudioSegment.from_file(file, format="m4a")
ma_audio.export("audios/audio.mp3", format="mp3")

os.remove(file)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
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
    torch_dtype=torch_dtype,
    use_fast=True,
    device=device,
)

summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

result = pipe("audios/audio.mp3", return_timestamps=True)
print("Transcription: "+ result["text"])
print(f"Transcription Length: {len(result["text"])}")
maxL = int(input("Summary max length"))
minL = int(input("Summary min length"))
summary = summarizer(result["text"], max_length=maxL, min_length=minL, do_sample=False)

print(summary[0]['summary_text'])