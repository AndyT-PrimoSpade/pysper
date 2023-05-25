import os
import whisper
from pyannote.audio import Pipeline
from utils import diarize_and_merge_text, write_results_to_txt_file, convert_txt_to_srt
from tqdm import tqdm


audio_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")
asr_model = whisper.load_model("medium.en")
asr_transcription = asr_model.transcribe("audio.wav", verbose=False)
for result in tqdm(range(1)):
    diarization_result = audio_pipeline("audio.wav", num_speakers=2)
diarized_text = diarize_and_merge_text(asr_transcription, diarization_result)
diarized_text_str = str(diarized_text)
merged_diarized_text = ' '.join(diarized_text_str)
segments = asr_transcription['segments']

write_results_to_txt_file(diarized_text, "output/result.txt")

convert_txt_to_srt("output/result.txt", "output/result.srt")
