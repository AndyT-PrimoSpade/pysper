from pyannote.core import Segment, Annotation, Timeline
import pysrt
import srt
import math
from tqdm import tqdm


def get_text_with_timestamp(asr_result):
    timestamp_texts = []
    for item in asr_result['segments']:
        start_time = item['start']
        end_time = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start_time, end_time), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, diarization_result):
    spk_text = []
    for seg, text in timestamp_texts:
        speaker = diarization_result.crop(seg).argmax()
        spk_text.append((seg, speaker, text))
    return spk_text


def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    speaker = text_cache[0][1]
    start_time = text_cache[0][0].start
    end_time = text_cache[-1][0].end
    return Segment(start_time, end_time), speaker, sentence


PUNCTUATION_SENTENCE_END = ['.', '?', '!']


def merge_sentence(spk_text):
    merged_spk_text = []
    previous_speaker = None
    text_cache = []
    for seg, speaker, text in tqdm(spk_text):
        if speaker != previous_speaker and previous_speaker is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, speaker, text)]
            previous_speaker = speaker

        elif text[-1] in PUNCTUATION_SENTENCE_END:
            text_cache.append((seg, speaker, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            previous_speaker = speaker
        else:
            text_cache.append((seg, speaker, text))
            previous_speaker = speaker
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text


def diarize_and_merge_text(asr_result, diarization_result):
    timestamp_texts = get_text_with_timestamp(asr_result)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed


def write_results_to_txt_file(final_result, file_name):
    with open(file_name, 'w') as fp:
        for seg, speaker, sentence in tqdm(final_result):
            line = f'{seg.start} / {seg.end} / {speaker} / {sentence}\n'
            fp.write(line)


# def write_whisper_to_srt_file(segments, file_name):
#     for segment in segments:
#         startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
#         endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
#         text = segment['text']
#         segmentId = segment['id']+1
#         segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] is ' ' else text}\n\n"
#         srtFilename = os.file_name.join("SrtFiles", f"VIDEO_FILENAME.srt")
#         with open(srtFilename, 'a', encoding='utf-8') as srtFile:
#             srtFile.write(segment)
#     return srtFilename

# def write_whisper_to_resultsrt_file(segments, file_name):
#     for seg, speaker, sentence in final_result:
#         line = f'{seg.start} {seg.end} {speaker} {sentence}\n'
#         with open(srtFilename, 'a', encoding='utf-8') as srtFile:
#             srt.write(line)

def convert_txt_to_srt(input_file, output_file):
    with open(input_file, "r") as input:
        lines = input.readlines()
    srt_subtitles = []
    for i, line in tqdm(enumerate(lines)):
        start = line.split("/")[0].strip()
        print(start)
        start = int(start)
        start = math.trunc(start)
        end = line.split("/")[1].strip()
        print(end)
        end = int(end)
        end = math.trunc(end)
        speaker = line.split("/")[2]
        main = line.split("/")[3]
        content = f'{speaker} -- {main}'
        subtitles = srt.Subtitle(i+1, start, end, content)
        srt_subtitles.append(subtitles)
    with open(output_file, "w") as output:
        for subtitle in srt_subtitles:
            output.write(subtitle.to_srt())

# This is to print the result on the console
# for seg, spk, sent in final_result:
#     line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
#     print(line)
