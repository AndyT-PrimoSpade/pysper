import os
import whisper
from pyannote.audio import Pipeline
from utils import diarize_and_merge_text, write_results_to_txt_file

audio_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")
asr_model = whisper.load_model("medium.en")
asr_transcription = asr_model.transcribe("audio.wav")
diarization_result = audio_pipeline("audio.wav")
diarized_text = diarize_and_merge_text(asr_transcription, diarization_result)
diarized_text_str = str(diarized_text)
merged_diarized_text = ' '.join(diarized_text_str)

write_results_to_txt_file(merged_diarized_text, "output/raw_result.txt")

write_results_to_txt_file(diarized_text, "output/result.txt")
