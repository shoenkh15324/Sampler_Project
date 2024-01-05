import numpy as np
import pyaudio
import pygame
from scipy import signal

# 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024

# FFT 설정
fft_size = 512
frequencies = np.fft.fftfreq(fft_size, 1.0 / RATE)
start_freq, end_freq = 200, 20000

# 주파수 대역을 250Hz 간격으로 설정
frequency_interval = 100
start_index = int(start_freq / frequency_interval)
end_index = int(end_freq / frequency_interval)

# High-pass filter parameters
cutoff_frequency = 500  # Adjust this cutoff frequency as needed
nyquist = 0.5 * RATE
normal_cutoff = cutoff_frequency / nyquist
b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)

# 데시벨 증폭 계수
db_amp = 1.5

# Pygame 초기화
pygame.init()

# 화면 설정
infoObject = pygame.display.Info()
screen_width = int(infoObject.current_w / 1.1)
screen_height = int(infoObject.current_w / 2.2)
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Audio Visualizer')

# 막대 개수, 간격, 너비 설정
num_bars = 50
bar_spacing = 5
bar_width = (screen_width - (num_bars - 1) * bar_spacing) // num_bars

x_positions = np.arange(num_bars) * (bar_width + bar_spacing)

# 초기 색상 설정 (데시벨에 따라 변경될 예정)
initial_color = (255, 255, 255)

# 막대 초기화 및 색상 설정
bars = [pygame.Rect(x, screen_height, bar_width, 0) for x in x_positions]
bar_colors = [initial_color] * num_bars  # 막대의 초기 색상 리스트

# 데시벨이 일정 수 이하일 때 바의 높이가 변하지 않도록 설정
min_decibel = 70  # 데시벨의 최소값 설정

# 바의 높이 변화를 부드럽게 만들기 위한 변수
smoothing_factor = 0.1  # 부드럽게 만들기 위한 계수

# 데시벨에 따라 색상을 변경하는 함수


def change_color_by_decibel(db):
    # 시작은 흰색, 데시벨에 따라 파란색으로 서서히 변하도록 설정
    r = int(min(max(db+10, 0), 255))  # 데시벨 값을 0에서 255 사이로 클립
    g = int(min(max(db+30, 0), 255))  # 데시벨 값을 0에서 255 사이로 클립
    b = int(min(max(db+60, 0), 255))  # 데시벨 값을 0에서 255 사이로 클립
    return (r, g, b)


# 오디오 스트림 열기
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    audio_data = stream.read(CHUNK)
    audio_buffer = np.frombuffer(audio_data, dtype=np.int16)
    
     # Apply high-pass filter
    audio_buffer = signal.filtfilt(b, a, audio_buffer)

    fft_buffer = np.abs(np.fft.fft(audio_buffer, fft_size)
                        [start_index:end_index + 1])
    db_spectrum = db_amp * 20 * np.log10(fft_buffer)

    for i, bar in enumerate(bars):
        if np.isinf(db_spectrum[i]) or np.isnan(db_spectrum[i]):
            target_bar_height = 0
        else:
            target_bar_height = int(db_spectrum[i] * 2)
            target_bar_height = max(0, target_bar_height)

        # 이전 높이와 새로운 높이 사이를 부드럽게 보간
        current_bar_height = bar.height
        interpolated_bar_height = current_bar_height + \
            smoothing_factor * (target_bar_height - current_bar_height)

        # 데시벨이 일정 수 이하인 경우 높이를 최소 높이로 고정
        if db_spectrum[i] < min_decibel:
            interpolated_bar_height = 30

        bar.height = interpolated_bar_height
        bar.y = screen_height - interpolated_bar_height

        # 데시벨 값에 따라 막대의 색상 변경
        bar_colors[i] = change_color_by_decibel(db_spectrum[i])

    screen.fill((0, 0, 0))
    for i, bar in enumerate(bars):
        pygame.draw.rect(screen, bar_colors[i], bar)

    pygame.display.flip()
    clock.tick(60)

stream.stop_stream()
stream.close()
p.terminate()

pygame.quit()
