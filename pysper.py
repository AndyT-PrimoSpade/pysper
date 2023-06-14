import os
import whisper
from pyannote.audio import Pipeline
from utils import diarize_and_merge_text, write_results_to_txt_file, convert_txt_to_srt, split_audio, whisper_txt_combine, combine_txt_file, adjust_cpu_usage, convert_m4a_to_wav, convert_audio_to_wav, has_file
from tqdm import tqdm
import psutil
import time
import datetime

print(datetime.datetime.now())

# psutil.cpu_percent(interval=1, percpu=False)
# adjust_cpu_usage()

audiofile_name = has_file()

filetype = ["m4a", "mp3", "mp4", "avi", "wav"]
for element in filetype:
    if element in audiofile_name:
        output_path = convert_audio_to_wav(audiofile_name)
        saved_name = audiofile_name.split("/")[1].split(".")[0]
        print(f"The {element} audio file has been convert to WAV format and saved to convert/{saved_name}.wav")

# audiofile_name = "MacTrade"

# convert_m4a_to_wav(f"input/{audiofile_name}.m4a", f"convert/{audiofile_name}.wav")

audiofile_name = audiofile_name.split("/")[1].split(".")[0]
main = f"convert/{audiofile_name}.wav"

audio_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")
asr_model = whisper.load_model("medium")

asr_transcription = asr_model.transcribe(main, verbose="True", language="chinese")

for result in tqdm(range(1)):
    diarization = audio_pipeline(main)

diarized_text = diarize_and_merge_text(asr_transcription, diarization)

write_results_to_txt_file(diarized_text, f"output/{audiofile_name}.txt")

convert_txt_to_srt(f"output/{audiofile_name}.txt", f"output/{audiofile_name}.srt")

print(datetime.datetime.now())

