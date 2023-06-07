import os
import whisper
from pyannote.audio import Pipeline
from utils import diarize_and_merge_text, write_results_to_txt_file, convert_txt_to_srt, split_audio, whisper_txt_combine, combine_txt_file, adjust_cpu_usage
from tqdm import tqdm
import psutil
import time
import datetime

print(datetime.datetime.now())

# psutil.cpu_percent(interval=1, percpu=False)
# adjust_cpu_usage()

audiofile_name = "MacTrade"

convert_m4a_to_wav(f"input/{audiofile_name}.m4a", f"convert/{audiofile_name}.wav")

main = f"convert/{audiofile_name}.wav"

audio_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")
asr_model = whisper.load_model("medium")
asr_transcription = asr_model.transcribe(main, verbose=False, language="en")

for result in tqdm(range(1)):
    diarization = audio_pipeline(main)

diarized_text = diarize_and_merge_text(asr_transcription, diarization)

write_results_to_txt_file(diarized_text, "output/result.txt")

convert_txt_to_srt("output/result.txt", "output/result.srt")

print(datetime.datetime.now())

