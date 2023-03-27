---
title: Simple LLaMA Finetuner
emoji: 🦙
colorFrom: yellow
colorTo: orange
sdk: gradio
app_file: main.py
pinned: false
---

# 🦙 Simple LLaMA Finetuner

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RandomityGuy/simple-llama-finetuner-kaggle/blob/master/Simple_LLaMA_FineTuner.ipynb)

Modification of Simple LLamA FineTuner to suit my needs.  

## Usage
```
main.py [trainingfile.json]
```
The trainingfile.json is a JSON file with a single JSON object of format
{
    "messages": [...]
}

## Acknowledgements

 - https://github.com/lxe/simple-llama-finetuner
 - https://github.com/zphang/minimal-llama/
 - https://github.com/tloen/alpaca-lora
 - https://github.com/huggingface/peft
 - https://huggingface.co/datasets/Anthropic/hh-rlhf

## License

MIT License

Copyright (c) 2023 Aleksey Smolenchuk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
