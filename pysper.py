import os
import whisper
from pyannote.audio import Pipeline
from utils import diarize_and_merge_text, write_results_to_txt_file, convert_txt_to_srt, split_audio, whisper_txt_combine, combine_txt_file, adjust_cpu_usage
from tqdm import tqdm
import psutil
import time
import datetime

# psutil.cpu_percent(interval=1, percpu=False)

print(datetime.datetime.now())
adjust_cpu_usage()

split_audio("audio.wav")
first = "audio/first_half.mp3"
second = "audio/second_half.mp3"

audio_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")
asr_model = whisper.load_model("medium")
first_asr_transcription = asr_model.transcribe(first, verbose=False, language="en")
second_asr_transcription = asr_model.transcribe(second, verbose=False, language="en")

whisper_text_1 = first_asr_transcription["text"]
whisper_text_2 = second_asr_transcription["text"]
whisper_txt_combine(whisper_text_1, whisper_text_2)

for result in tqdm(range(1)):
    first_diz = audio_pipeline(first)
    second_diz = audio_pipeline(second)

diarized_text_1 = diarize_and_merge_text(first_asr_transcription, first_diz)
diarized_text_2 = diarize_and_merge_text(second_asr_transcription, second_diz)

write_results_to_txt_file(diarized_text_1, "output/result_1.txt")
write_results_to_txt_file(diarized_text_1, "output/result_2.txt")
combine_txt_file("output/result_1.txt", "output/result_2.txt", "output/result.txt")
convert_txt_to_srt("output/result.txt", "output/result.srt")

if os.path.exists("output/result_1.txt"):
    os.remove("output/result_1.txt")
if os.path.exists("output/result_2.txt"):
    os.remove("output/result_2.txt")

print(datetime.datetime.now())

