import pickle
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

from enum import Enum
from image import *
from text import *

sd.default.samplerate = fs = 44100  # sampling frequency -> 44.1 KHz
sd.default.channels = 1


def play(audio):
    sd.play(audio), sd.wait()


def record(seconds: float):
    recording, _ = sd.rec(seconds * fs), sd.wait()
    return recording


def play_record(audio):
    pause_length = 2.5  # seconds
    delay_length = 0.05  # seconds
    pause = np.zeros(int(pause_length * fs))
    delay = np.zeros(int(delay_length * fs))
    audio_with_pause = np.concatenate((pause, audio, delay))
    recording, _ = sd.playrec(audio_with_pause), sd.wait()
    return recording[int((pause_length + delay_length) * fs):, 0]


def freq_audio(frequency: int, seconds: float):
    t = np.arange(0, seconds, 1 / fs)  # x-axis
    audio = np.sin((2 * np.pi * frequency) * t)  # y-axis
    return audio


def composite_freq_audio(frequencies: list, seconds: float):
    audio = np.zeros(int(seconds * fs))
    for freq in frequencies:
        audio += freq_audio(freq, seconds)
    return audio / len(frequencies)


CHUNK_DURATION = 0.1  # seconds
CHUNK_LENGTH = int(fs * CHUNK_DURATION)
START = 1000
STEPS = 100


def chunk_freq_audio(frequencies: list):
    audio = np.concatenate(
        tuple([freq_audio(freq, CHUNK_DURATION) for freq in frequencies])
    )
    return audio


def break_chunks(audio: np.ndarray):
    # length -> fs * duration | fs * chunks * CHUNK_DURATION | chunks * CHUNK_LENGTH
    length = len(audio)
    chunks = length // CHUNK_LENGTH
    for idx in range(chunks):
        start = idx * CHUNK_LENGTH
        end = start + CHUNK_LENGTH
        yield audio[start: end]


def break_further(chunk: np.ndarray):
    # failed experiment
    length = len(chunk)
    point_a = length // 3
    point_b = 2 * point_a
    return [chunk[: point_a], chunk[point_a: point_b], chunk[point_b:]]


def decode_chunk(chunk: np.ndarray):
    w = np.abs(np.fft.fft(chunk))
    freqs = np.fft.fftfreq(len(chunk), d=1 / fs)
    peak_coeff = np.argmax(np.abs(w))
    peak_freq = freqs[peak_coeff]
    return abs(peak_freq)


def decode(audio):
    signal = []
    for chunk in break_chunks(audio):
        # failed experiment
        # # further break into parts and take majority of all frequencies
        # candidate = {}

        # for part in break_further(chunk):
        #     freq = decode_chunk(part)
        #     data = int(round((freq - START) / STEPS, 3))
        #     if data in candidate:
        #         candidate[data] += 1
        #     else:
        #         candidate[data] = 1
        # print(candidate)
        # signal.append(max(candidate, key=candidate.get))
        # # time.sleep(1); play(chunk)

        freq = int(decode_chunk(chunk))
        data = int(round((freq - START) / STEPS))
        signal.append(data)

    return signal


def encode(data: list):
    frequencies = []
    for value in data:
        frequencies.append(START + value * STEPS)
    return frequencies


class Type(Enum):
    IMAGE = 1
    TEXT = 2
    ARRAY = 3
    DEMO = 4


CONFIGURATION = Type.TEXT
PLAY = True
RECORD_FILE = './recorded_audio'


image = read_pgm('image_input.pgm')
txt = read_txt('text_input.txt')


data_switch = {
    Type.TEXT: txt_to_data(txt),
    Type.IMAGE: image_to_data(image),
    Type.ARRAY:  np.random.randint(0, 16, 10, dtype=np.int16),
    Type.DEMO: [4, 2, 5, 10, 9, 7, 2, 11, 3, 9]
}

data = data_switch.get(CONFIGURATION, np.ndarray(1, dtype=np.int16))
audio = chunk_freq_audio(encode(data))
audio_fft = np.abs(np.fft.fft(audio))

signal_transmitted = decode(audio)
print('Signal Transmitted :', signal_transmitted)

recorded_audio = play_record(audio) if PLAY else np.load(RECORD_FILE + '.npy')
if PLAY and CONFIGURATION == Type.DEMO:
    np.save(RECORD_FILE, recorded_audio)
recorded_audio_fft = np.abs(np.fft.fft(recorded_audio))


signal_received = decode(recorded_audio)
print('Signal Received :', signal_received)

if CONFIGURATION == Type.TEXT:
    write_txt('text_output.txt',  data_to_txt(signal_received))
elif CONFIGURATION == Type.IMAGE:
    write_pgm('image_output.pgm',  data_to_image(signal_received))

PLOT = False
if PLOT:
    plt.plot(audio, alpha=0.4)
    plt.plot(recorded_audio, alpha=0.5)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.savefig('recorded_audio.png', dpi=500)
    plt.clf()

    plt.plot(audio_fft, alpha=0.5)
    plt.xlabel('Frequency')
    plt.savefig('audio_fft.png', dpi=500)
    plt.clf()

    plt.plot(recorded_audio_fft, alpha=0.5)
    plt.xlabel('Frequency')
    plt.savefig('recorded_audio_fft.png', dpi=500)
    plt.clf()

    plt.plot(signal_transmitted, alpha=0.5)
    plt.plot(range(1, len(signal_received) + 1), signal_received, alpha=0.5)
    plt.ylim(0, 16)
    plt.xlabel('Data')
    plt.ylabel('Index')
    plt.savefig('data.png', dpi=500)
    plt.clf()

