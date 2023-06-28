from pyannote.core import Segment, Annotation, Timeline
import math
from tqdm import tqdm
from datetime import datetime, timedelta
import os
from pydub import AudioSegment
import psutil
import time
import ffmpeg
import subprocess
import io
import ctypes
import tkinter as tk

def has_file():
    while True:
        audio_filename = input("Audio File Name: ")
        audiofile_name = f"../input/{audio_filename}"
        if not os.path.exists(audiofile_name):
            print("Invalid file name, please enter again")
            continue
        return audiofile_name

def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res['segments']:
        start = item['start']
        end = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text


def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence


PUNC_SENT_END = ['.', '?', '!']


def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text


def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed


def write_to_txt(spk_sent, file):
    with open(file, 'w') as fp:
        for seg, spk, sentence in spk_sent:
            line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n'
            fp.write(line)

def convert_txt_to_srt(input_file, output_file):
    with io.open(input_file, "r", encoding="utf-8") as input:
        lines = input.readlines()
    if os.path.exists(output_file):
        os.remove(output_file)
    else:
        pass
    srt_subtitles = []
    for i, line in tqdm(enumerate(lines)):
        start = line.split("/")[0].strip()
        start = float(start)
        start = int(start)
        start = timedelta(seconds = start)
        end = line.split("/")[1].strip()
        end = float(end)
        end = int(end)
        end = timedelta(seconds = end)
        speaker = line.split("/")[2].strip()
        main = line.split("/")[3].strip()
        content = f'{speaker} -- {main}'
        subtitles = f'{i+1}\n{start} --> {end}\n{content}\n\n'
        srt_subtitles.append(subtitles)
        with io.open(output_file, "a", encoding="utf-8") as output:
            output.write(subtitles)


def split_audio(fileName):
    audio_name = fileName
    audio_half = audio_name.split(".")[0]
    audio = AudioSegment.from_file(audio_name)

    duration = len(audio)

    half_point = int(duration / 2)
    extra_duration = duration % 2

    first_half = audio[:half_point]
    second_half = audio[half_point + extra_duration:]

    first_half.export(f"../audio/first_half.mp3")

    second_half.export(f"../audio/second_half.mp3")

def combine_txt_file(file1, file2, output):
    with io.open(file1, "r", encoding="utf-8") as f1:
        content1 = f1.read()
    with io.open(file2, "r", encoding="utf-8") as f2:
        content2 = f2.read()
    with io.open(output, "w", encoding="utf-8") as f3:
        f3.write(content1 + content2)

    if __name__ == "__main__":
        file1 = os.path.join(os.getcwd(), "file1.txt")
        file2 = os.path.join(os.getcwd(), "file2.txt")
        output = os.path.join(os.getcwd(), "output.txt")

def whisper_txt_combine(input_1, input_2):    
    with io.open("../output/whisper1.txt", 'w', encoding="utf-8") as text:
        text.write(input_1)
    with io.open("../tput/whisper2.txt", 'w', encoding="utf-8") as text:
        text.write(input_2)
    combine_txt_file("../output/whisper1.txt", "../output/whisper2.txt", "../output/whisper.txt")
    if os.path.exists("../output/whisper1.txt"):
        os.remove("../output/whisper1.txt")
    if os.path.exists("../output/whisper2.txt"):
        os.remove("../output/whisper2.txt")

def adjust_cpu_usage():
    cpu_limit = 20
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > cpu_limit:
            time.sleep(0.1)
        else:
            break

def convert_m4a_to_wav(input_file, output_file):
    ffmpeg.input(input_file).output(output_file, format='wav').run()
    print("Conversion Completed!")

def convert_audio_to_wav(file_path):
    filenaming = file_path.split("/")[-1].split(".")[0]
    ffmpeg_command = ["ffmpeg", "-i", file_path, "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2", "-f", "wav", f"../convert/{filenaming}.wav"]
    subprocess.run(ffmpeg_command)

def clear_cmd():
    os.system("cls")

def min_console():
    console_handle = ctypes.windll.kernel32.GetConsoleWindow()
    ctypes.windll.user32.ShowWindow(console_handle, 0)

def finish_popup(duration):
    duration = duration.split(".")[0]
    popup = tk.Tk()
    popup.title(f"Transcript Completed.")
    popup.geometry("350x100")
    label = tk.Label(popup, text=f"Transcript has Completed Running. Total Time = {duration}")
    label.pack() 
    popup.protocol("WM_DELETE_WINDOW", lambda: popup.destroy())
    popup.mainloop()

# This is to print the result on the console
# for seg, spk, sent in final_result:
#     line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
#     print(line)
