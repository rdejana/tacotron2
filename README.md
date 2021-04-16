# tacotron2
Getting simple tacotron2 example running on Jetson NX

First, make sure pytorch is installed.

This is our goal: https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/

This will require the following:
```
numpy scipy librosa unidecode inflect librosa
```

librosa is a bit of a challenge to install and you'll need to do the following before installing:
```
sudo mv /usr/include/tbb/tbb.h /usr/include/tbb/tbb.bak
```
This is from https://forums.developer.nvidia.com/t/instll-python-packages-librosa-and-llvm-on-jetson-nano-developer-kit-problem/74543/7

Now install LLVM.
```
sudo apt-get install llvm-10 llvm-10-dev
```
Before installing the dependencies, run the command:
```
export LLVM_CONFIG=/usr/bin/llvm-config-10
```
Now let's try installing our python resources:
```
pip3 install numpy scipy librosa unidecode inflect librosa
```

Let's put the file back we moved at the start: 
```
sudo mv /usr/include/tbb/tbb.bak /usr/include/tbb/tbb.h
```

We are now ready to try running the example.  Create a file named `test.py` with the following content:
```
import torch
import numpy as np
from scipy.io.wavfile import write

tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')

tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

text = "hello world, I missed you"

# preprocessing
sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

# run the models
with torch.no_grad():
    _, mel, _, _ = tacotron2.infer(sequence)
    audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050

write("audio.wav", rate, audio_numpy)
```

Then run with the command `python3 test.py`.

If you get an error similar to 
```
ImportError: /home/rdejana/.local/lib/python3.6/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
```
you will need to preload the library.  Using the path to your library, run the following command:
```
export LD_PRELOAD=/home/rdejana/.local/lib/python3.6/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
```
and rerun `python3 test.py`.

When completed, you should see the file `audio.wav`.  You can play on your nx with headphones or copy to another machine.



Links I used:
- https://forums.developer.nvidia.com/t/cuda-toolkit-version-problem-when-trying-to-run-a-python-script-on-gpu-through-numbas-jit-cuda-modules-on-agx-xavier/129469/5
- https://forums.developer.nvidia.com/t/instll-python-packages-librosa-and-llvm-on-jetson-nano-developer-kit-problem/74543/7
- https://stackoverflow.com/questions/61065541/importerror-usr-lib-aarch64-linux-gnu-libgomp-so-1-cannot-allocate-memory-in
