# <span style="color:green"> __PyWhisper Installation__
- Install python 3.10
- Install pip
- Install ffmpeg, windows - download file at ffmpeg.org. Click on Windows EXE file and click on Windows builds from gyan.dev. Once at gyan.dev scroll down and download ffmpeg-git-full.7z. <br /> Extract zip and rename to ffmpeg then move to root folder C://. RUN CMD in admin set the eviroment path by running `setx /m PATH "C:\Users/user_name/ffmpeg\bin;%PATH%"`. <br />
Restart PC and run `ffmpeg -version` to check if install.
- Install OpenAI Whisper via `pip install git+https://github.com/openai/whisper.git`
- Update Whisper via `pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git`
- Install Rust Tools via `pip install setuptools-rust`
- Install Pyannote via `pip install git+https://github.com/pyannote/pyannote-audio.git`
- Change setuptools via `pip install setuptools==59.5.0`
- Download project file from `git clone https://github.com/AndyT-PrimoSpade/pysper.git`

# <span style="color:green"> __Running Script__
- Update the use_auth_token by register via huggingface for the token
- CMD/PowerShell cd to the dir and run `python pysper.py shell=True`
- Under asr_transcription and diarized_text_str change the name to audio file name.
- Under diarized_text_str for num_speakers can set if speaker number is known.