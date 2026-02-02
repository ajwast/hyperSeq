# HyperSeq

## Introduction

HyperSeq is an experimental prototyping project in using neural network architectures for generative MIDI sequencing.  It is an extention of the Deep Steps concept to be used as a scaleable music performance tool. The neural network models are intended to be computationally light enough to run on an embedded system and work in real-time.

## Installation

HyperSeq is a project using Python and Pytorch. Clone the repo and install the dependencies via pip

``` bash
git clone https://github.com/ajwast/hyperSeq.git
```

Dependencies are basically torch and python-rtmidi. As this is intended to run on rPi5, it uses the CPU version of torch. If you want this too then use the requirements.txt...

``` bash
cd path/to/project
pip install requirements.txt
```

Otherwise get them separately

``` bash
pip install python-rtmidi torch
```

Start your environment and the program.

``` bash
source env/bin/activate
python3 src/main.py
```

You will be prompted to choose MIDI in and out ports from those available on your system. The models will then train on some pre-processed data. A simple Tkinter GUI will give some visual feedback. Send MIDI clock to the MIDI input to run the sequencer.