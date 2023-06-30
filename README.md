# <span style="color:green"> __Pysper Installation__
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
- CMD/PowerShell cd to the dir and run `python pysper.py`
- Under asr_transcription and diarized_text_str change the name to audio file name.
- Under diarized_text_str for num_speakers can set if speaker number is known.

# <span style="color:green"> __Install Via Pip__
- Uninstall torch torchvision torchaudio using pip
- https://pytorch.org/get-started/locally/ `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`

# <span style="color:green"> __Install Via Conda__
- Install Anaconda - use base env (Anacoda Powershell Prompt)
- conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
- pip install the rest

- This is for the bat file to run Anacoda Powershell prompt
`@echo off call C:\Anaconda3\Scripts\activate.bat cd C:\path\to\your\code python your_code.py`


# <span style="color:green"> __To Runn Offline__
- Install HF CLI `pip install huggingface_hub`
- Run CLI to login `huggingface-cli login`
- Save Cred to local `huggingface-cli login --token your_token`
- Run Pysper as per normal to download model
- Check cred is in local
