from pyannote.core import Segment, Annotation, Timeline


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
    for seg, speaker, text in spk_text:
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
        for seg, speaker, sentence in final_result:
            line = f'{seg.start:.2f} {seg.end:.2f} {speaker} {sentence}\n'
            fp.write(line)

# This is to print the result on the console
# for seg, spk, sent in final_result:
#     line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
#     print(line)
