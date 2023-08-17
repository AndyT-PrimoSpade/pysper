import os
import whisper
from pyannote.audio import Pipeline
from utils import diarize_and_merge_text, write_results_to_txt_file, convert_txt_to_srt, adjust_cpu_usage, convert_audio_to_wav, clear_cmd, clear_purge
from tqdm import tqdm
import psutil
import time
import datetime
import sys
import torch



start = datetime.datetime.now()
psutil.cpu_percent(interval=1, percpu=False)
device = torch.device("cuda:0")
# device = torch.device("cpu")

audio_filename = input("Audio File Name: ")
audiofile_name = f"../input/{audio_filename}"
filetype = ["m4a", "mp3", "mp4", "avi"]
for element in filetype:
    if element in audiofile_name:
        output_path = convert_audio_to_wav(audiofile_name)
        saved_name = audiofile_name.split("/")[-1].split(".")[0]
        print(f"The {element} audio file has been convert to WAV format and saved to convert/{saved_name}.wav")
audiofile_name = audiofile_name.split("/")[-1].split(".")[0]
main = f"../convert/{audiofile_name}.wav"
with open("output.txt", "w") as f:
    sys.stdout = f
    audio_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=None).to(device)
    sys.stdout = sys.__stdout__
if os.path.exists("output.txt"):
    os.remove("output.txt")
asr_model = whisper.load_model("medium").to(device)
asr_model.to(torch.device("cuda:0"))
audio_pipeline.to(torch.device("cuda:0"))
asr_transcription = asr_model.transcribe(main, verbose=False, language="en")
for result in tqdm(range(1)):
    diarization = audio_pipeline(main)
diarized_text = diarize_and_merge_text(asr_transcription, diarization)
write_results_to_txt_file(diarized_text, f"../output/{audiofile_name}.txt")
convert_txt_to_srt(f"../output/{audiofile_name}.txt", f"../output/{audiofile_name}.srt")
end = datetime.datetime.now()
print(end-start)
clear_purge()


