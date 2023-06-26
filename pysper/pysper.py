import os
import whisper
from pyannote.audio import Pipeline
from utils import diarize_and_merge_text, write_results_to_txt_file, convert_txt_to_srt, split_audio, whisper_txt_combine, combine_txt_file, adjust_cpu_usage, convert_m4a_to_wav, convert_audio_to_wav, clear_cmd, min_console, finish_popup
from tqdm import tqdm
import psutil
import time
import datetime
import sys
import torch

print(datetime.datetime.now())
start = datetime.datetime.now()
psutil.cpu_percent(interval=1, percpu=False)
# device = torch.device("cuda:0")
device = torch.device("cpu")
# adjust_cpu_usage()
audio_filename = input("Audio File Name: ")
audiofile_name = f"../input/{audio_filename}"
clear_cmd()
filetype = ["m4a", "mp3", "mp4", "avi"]
for element in filetype:
    if element in audiofile_name:
        output_path = convert_audio_to_wav(audiofile_name)
        saved_name = audiofile_name.split("/")[-1].split(".")[0]
        print(f"The {element} audio file has been convert to WAV format and saved to convert/{saved_name}.wav")
clear_cmd()
# min_console()
audiofile_name = audiofile_name.split("/")[-1].split(".")[0]
main = f"../convert/{audiofile_name}.wav"
clear_cmd()
audio_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")
clear_cmd()
asr_model = whisper.load_model("medium").to(device)
# asr_model = whisper.load_model("medium")
# asr_model.to(torch.device("gpu"))
# audio_pipeline.to(torch.device("gpu"))
clear_cmd()
asr_transcription = asr_model.transcribe(main, verbose=False, language="en")
clear_cmd()
for result in tqdm(range(1)):
    diarization = audio_pipeline(main)
clear_cmd()
diarized_text = diarize_and_merge_text(asr_transcription, diarization)

write_results_to_txt_file(diarized_text, f"../output/{audiofile_name}.txt")

convert_txt_to_srt(f"../output/{audiofile_name}.txt", f"../output/{audiofile_name}.srt")

print(datetime.datetime.now())
end = datetime.datetime.now()

# duration = end - start
# finish_popup(duration)

