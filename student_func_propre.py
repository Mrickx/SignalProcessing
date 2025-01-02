# %% [markdown]
# # Signal Processing Project: real-time sound localisation

# %% [markdown]
# ## 1 Offline system
# ### 1.1 Data generation and dataset

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from Graph import graph
import re 
def create_sine_wave(f, A, fs, N):
    
    # your code here #

    t=np.arange(0,N/fs,1/fs)
   
    return A*np.sin(2*np.pi*f*t)

# call and test your function here #
fs = 44100
N = 8000
freq = 20
amplitude = 4

sin=create_sine_wave(freq,amplitude,fs,N)
plt.plot(sin)
graph("Signal sinusoïdale de fréquence 20 Hz","Echantillon","Amplitude")
plt.show()

# %%
from glob import glob
import scipy.io.wavfile as wf
import audiofile 


def read_wavefile(path):

    # your code here #
    signal_audio, sampling_rate = audiofile.read(path)

    return signal_audio,sampling_rate

# call and test your function here #
LocateClaps = "LocateClaps"
files = glob(f"{LocateClaps}/*.wav")
# Fonction de clé de tri
def sorting_key(f):
    match = re.search(r'M(\d+)_(\d+)', f)
    if match:
        group = int(match.group(1))  # Numéro après 'M'
        angle1 = int(match.group(2))  # Angle
        return (group, angle1)
    return (float('inf'), float('inf'))  # Pour gérer les fichiers sans correspondance

# Triage par groupe (M1, M2, etc.) et angle croissant
files = sorted(files, key=sorting_key)

#print(files) 
x=len(files)
# Chemin vers le fichier spécifique
all_signals = []

# Lecture et traitement de chaque fichier
for file in files:
    signal_audio, sampling_rate = read_wavefile(file)
    all_signals.append(signal_audio)  # Ajouter le signal à la liste
plt.plot(all_signals[0])
graph({files[0]},"Temps","Amplitude")
plt.show()

# Affiche le nombre total de fichiers traités
print(f"{len(files)} fichiers ont été traités.")

# %% [markdown]
# ### 1.2 Buffering

# %%
from collections import deque

def create_ringbuffer(maxlen):
    
   # your code here #
    out = deque(maxlen=maxlen)

    return out

# call and test your function here #
stride = 0
maxlen = 750

# reading your signal as a stream:
# reading your signal as a stream:
my_buffer = create_ringbuffer(maxlen)
for i, sample in enumerate(signal_audio):
    # your code here #
    my_buffer.append(sample)
    

# %% [markdown]
# ### 1.3 Pre-processing
# #### 1.3.1 Normalisation

# %%
def normalise(s):
    
    # your code here #
    max_val=np.max(np.abs(s))
    if max_val == 0:
        return s

    out = s/max_val 

    return out

# call and test your function here #
sin_normalise = normalise(sin)
plt.plot(sin, label="Original")
plt.plot(sin_normalise, label="Normalisé")
graph("Comparaison du signal original et normalisé","Echantillon","Amplitude")
plt.legend()
plt.show()

# %% [markdown]
# #### 1.3.2 Downsampling

# %%
## 1 - spectral analysis via spectrogram

specific_file = "LocateClaps\\M1_0.wav"#J'ai besoin signal audio 
signal_audio, sampling_rate = read_wavefile(specific_file)

plt.specgram( signal_audio, Fs= sampling_rate )
plt.title("Spectrogram")
plt.show()

## 2 - Anti-aliasing filter synthesis
def create_filter_cheby(wp, ws, gpass, gstop, fs):

    # your code here #
    N,Wn = signal.cheb1ord(wp, ws, gpass,gstop,fs=fs)
    
    B, A = signal.cheby1(N,gpass,wp,'low',fs=fs)
    return B, A

def create_filter_cauer(wp, ws, gpass, gstop, fs):

    # your code here #
    N,Wn = signal.ellipord(wp,ws,gpass,gstop,fs=fs)

    B, A = signal.ellip(N,gpass,gstop,wp,'lowpass',fs=fs)

    return B, A
    

## 3 - Decimation
def downsampling(sig, B, A, M):

    # your code here #
    signal_filtered = signal.lfilter(B,A,sig)

    signal_decimated = signal_filtered[::M]
     
    return signal_decimated
    


# call and test your function here
sin1=create_sine_wave(8500,1000,fs,N)
sin2=create_sine_wave(7500,20,fs,N)
M=3
sin_test = sin1 + sin2
B_cauer,A_cauer = create_filter_cauer(16000,20000,5,40,fs)
sin_test_downsampled = downsampling(sin_test,B_cauer,A_cauer,M)

w,h=signal.freqz(sin_test_downsampled,fs=fs)
plt.plot(w,abs(h))

plt.show()

# %% [markdown]
# ### 1.4 Cross-correlation

# %%
## 1.4
import scipy.signal as sc
import numpy as np

def fftxcorr(in1, in2):
    
    # your code here #
    n = len(in1) + len(in2) - 1
    
    FFT1 = np.fft.fft(in1,n=n)
    FFT2 = np.fft.fft(in2,n=n)
    
    corr = np.fft.ifft(FFT1 * np.conj(FFT2))
    
    corr = np.fft.fftshift(corr.real)

    out=corr

    return out
    
    
# call and test your function here #

xcorr_fftconv = sc.fftconvolve(sin1, sin2[::-1], 'full') # [::-1] flips the signal but you can also use np.flip()

xcorr = fftxcorr(sin1,sin2)

n_conv = np.arange(-(len(sin1) - 1), len(sin1))
n_fft = np.arange(-len(sin1) + 1, len(sin1))

plt.figure(figsize=(10, 4))
plt.plot(n_fft, xcorr, label='Autocorrelation (fftxcorr)', color='blue')
plt.plot(n_conv, xcorr_fftconv, label='Autocorrelation (fftconvolve)', color='orange', linestyle='dashed')
plt.title("Comparison of Autocorrelation Methods")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ### 1.5 Localisation
# #### 1.5.1 TDOA

# %%
def TDOA(xcorr):
    
    # your code here #
    max_index = np.argmax(np.abs(xcorr))
    sample_offset = max_index - (len(xcorr) / 2)
    time_delay=sample_offset/fs
    
    return time_delay
    

# %% [markdown]
# #### 1.5.2 Equation system

# %%
from scipy.optimize import root

# mic coordinates in meters
MICS = [{'x': 0, 'y': 0.0487}, {'x': 0.0425, 'y': -0.025}, {'x': -0.0425, 'y': -0.025}] 

def equations(p, deltas):
    v = 343
    x, y = p
    alpha = np.arctan2((MICS[1]['y'] - MICS[0]['y']), (MICS[1]['x'] - MICS[0]['x']))
    beta = np.arctan2((MICS[2]['y'] - MICS[0]['y']), (MICS[2]['x'] - MICS[0]['x']))
    
    eq1 = v*deltas[0] - (np.sqrt((MICS[1]['x'] - MICS[0]['x'])**2 + (MICS[1]['y'] - MICS[0]['y'])**2) * np.sqrt((x)**2 + (y)**2) * np.cos(alpha-np.arctan2(y, x)))
    eq2 = v*deltas[1] - (np.sqrt((MICS[2]['x'] - MICS[0]['x'])**2 + (MICS[2]['y'] - MICS[0]['y'])**2) * np.sqrt((x)**2 + (y)**2) * np.cos(beta-np.arctan2(y, x)))
    return (eq1, eq2)
    
def localize_sound(deltas):

    sol = root(equations, [0, 0], (deltas), tol=10)
    return sol.x

def source_angle(coordinates):
    
    # your code here
    x = coordinates[0]
    y = coordinates[1]
    
    out = np.arctan(y/x)   #vérifier formule, pourquoi x et pas y
    out = np.degrees(out)
    
    if(x > 0 and y> 0 ):
        return out
    if(x < 0 and y> 0 ):
        return out + 180
    if(x < 0 and y < 0 ):
        return out + 180
    if(x > 0 and y < 0 ):
        return out + 360
    

# call and test your function here #

all_signals_post = ['']*len(files)
for i in range(len(files)):
    a = normalise(all_signals[i])
    all_signals_post[i] = downsampling(a,B_cauer,A_cauer,M)
possible_angle = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
threshold = 10  #choix arbitraire
n=0
xcorr12 = fftxcorr (all_signals_post[n],all_signals_post[n+12])
xcorr13 = fftxcorr (all_signals_post[n],all_signals_post[n+24])
deltas = np.array([TDOA(xcorr12),TDOA(xcorr13)])
xy=localize_sound(deltas)
pred_angle=source_angle(xy)
print(f"On essaie de prédire l'angle de {possible_angle[0]} degré")
print(f"Angle prédit : {pred_angle}")


    


# %% [markdown]
# ### 1.6 System accuracy and speed

# %%
## 1.6.1
def accuracy(pred_angle, gt_angle, threshold):
    
    # your code here #
    if np.abs(pred_angle-gt_angle)<threshold:
        return True
    else:
        return False
    

## 1.6.2
possible_angle = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
for angle in possible_angle:
    for f in files:
        if f'_{angle}.' in f:
            mic = f.split('/')[-1].split('_')[0] #if '/' does not work, use "\\" (windows notation)
            
# call and test your function here #
def estimation_angle(all_signals_post,k):
    xcorr12 = fftxcorr (all_signals_post[k],all_signals_post[k+12])
    xcorr13 = fftxcorr (all_signals_post[k],all_signals_post[k+24])
    deltas = np.array([TDOA(xcorr12),TDOA(xcorr13)])
    xy=localize_sound(deltas)
    pred_angle=source_angle(xy)
    print(f"Angle prédit : {pred_angle}")
    return(pred_angle)

for i,angle in enumerate(possible_angle):
    index=np.zeros(3).astype(int)
    i=0
    for f in files:
        if f'_{angle}.' in f:
            index[i]=files.index(f)
            i=+1
    angle2=estimation_angle(all_signals_post,index[0])
    print(f"Vraie valeur de l'angle : {angle}")
    print(accuracy(angle2,angle,10))

## 1.6.3
from time import time_ns, sleep

def func_example(a, b):
    return a*b

def time_delay(func, args):
    start_time = time_ns()
    out = func(*args)
    end_time = time_ns()
    print(f"{func.__name__} in {end_time - start_time} ns")
    return out

product = time_delay(func_example, [2, 10])

# call and test your previous functions here #

normalisés=time_delay(normalise,[all_signals[i]])
a=time_delay(downsampling,[normalisés,B_cauer,A_cauer,M])

print("Time delay lorsque les signaux sont downsamplés")
xcorr1=time_delay(fftxcorr,[all_signals_post[n],all_signals_post[n+12]])
xcorr2=time_delay(sc.fftconvolve,[all_signals_post[n], all_signals_post[n+24]])
tdoa1=time_delay(TDOA,[xcorr1])
tdoa2=TDOA([xcorr2])
xy=time_delay(localize_sound,[np.array([tdoa1,tdoa2])])
angle=time_delay(source_angle,[xy])


print("Time delay lorsqu'ils ne sont pas downsamplés")
all_signals_normalisés = ['']*len(files)
for i in range(len(files)):
    all_signals_normalisés[i] = normalise(all_signals[i])
  
xcorr3=time_delay(fftxcorr,[all_signals_normalisés[n],all_signals_normalisés[n+12]])
xcorr4=time_delay(sc.fftconvolve,[all_signals_normalisés[n], all_signals_normalisés[n+24]])
tdoa3=time_delay(TDOA,[xcorr3])
tdoa4=TDOA([xcorr4])
xy2=time_delay(localize_sound,[np.array([tdoa3,tdoa4])])
angle=time_delay(source_angle,[xy2])

# %% [markdown]
# ## 2 Real-time localisation

# %% [markdown]
# ### 2.1 Research one Raspberry Pi application

# %% [markdown]
# ### 2.2 Data acquisition and processing

# %%
#### Callback 
import pyaudio

RESPEAKER_CHANNELS = 8
BUFFERS = []

def callback(in_data, frame_count, time_info, flag):
    global BUFFERS
    data = np.frombuffer(in_data, dtype=np.int16)
    BUFFERS[0].extend(data[0::RESPEAKER_CHANNELS])
    BUFFERS[1].extend(data[2::RESPEAKER_CHANNELS])
    BUFFERS[2].extend(data[4::RESPEAKER_CHANNELS])
    return (None, pyaudio.paContinue)

#### Stream management

RATE = 44100
RESPEAKER_WIDTH = 2
CHUNK_SIZE = 2048

def init_stream():
    print("========= Stream opened =========")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)

        if device_info['maxInputChannels'] == 8:
            INDEX = i
            break

        if i == p.get_device_count()-1:
            # Sound card not found
            raise OSError('Invalid number of channels')

    stream = p.open(rate=RATE, channels=RESPEAKER_CHANNELS, format=p.get_format_from_width(RESPEAKER_WIDTH), input=True, input_device_index=INDEX,
                    frames_per_buffer=CHUNK_SIZE, stream_callback=callback)

    return stream



def close_stream(stream):
    print("========= Stream closed =========")
    stream.stop_stream()
    stream.close()

#### Detection and visual feedback
def detection(stream):
    global BUFFERS, pixel_ring
    
    if stream.is_active():
        print("========= Recording =========")

    while stream.is_active():
        try:
            if len(BUFFERS[0]) > CHUNK_SIZE:
                st = time_ns()
                deltas = [TDOA(fftxcorr(BUFFERS[0], BUFFERS[1])), TDOA(fftxcorr(BUFFERS[0], BUFFERS[2]))] 

                x, y = localize_sound(deltas)
                hyp = np.sqrt(x**2+y**2)
                
                ang_cos = round(np.arccos(x/hyp)*180/np.pi, 2)
                ang_sin = round(np.arcsin(y/hyp)*180/np.pi, 2)

                if ang_cos == ang_sin:
                    ang = ang_cos
                else:
                    ang = np.max([ang_cos, ang_sin])
                    if ang_cos < 0 or ang_sin < 0:
                        ang *= -1
                ang *= -1

                print((time_ns() - st)/1e9, ang)

                print(np.max(BUFFERS, axis=-1))

                if (np.max(BUFFERS, axis=-1) > 3000).any():
                    pixel_ring.wakeup(ang)
                else:
                    pixel_ring.off()

                sleep(0.5)

        except KeyboardInterrupt:
            print("========= Recording stopped =========")
            break

#### Launch detection
from pixel_ring.apa102_pixel_ring import PixelRing
from gpiozero import LED


USED_CHANNELS = 3


power = LED(5)
power.on()

pixel_ring = PixelRing(pattern='soundloc')

pixel_ring.set_brightness(10)

for i in range(USED_CHANNELS):
    BUFFERS.append(create_ringbuffer(3 * CHUNK_SIZE))
    
stream = init_stream()

while True:
    try:
        detection(stream)
        sleep(0.5)
    except KeyboardInterrupt:
        break

close_stream(stream)

power.off()

