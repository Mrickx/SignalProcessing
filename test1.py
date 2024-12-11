# %% [markdown]
# # Signal Processing Project: real-time sound localisation
#!!!!!!!!!!!!!!!!!ATTENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 #!!! Ne pas appeler les variables signal car il peut confonde avec signal. de scipy!!!
 #J'ai donc remplacer tous les "signal" --> "signal_audio"
#!!!!!!!!!!!!!!!!Attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
 
# %% [markdown]
# ## 1 Offline system
# ### 1.1 Data generation and dataset
#ici

# %%
import numpy as np
import matplotlib.pyplot as plt
from Graph import graph
import os 
from scipy import signal
from scipy.signal import lfilter

def create_sine_wave(f, A, fs, N):
    
   t=np.arange(0,N/fs,1/fs)
   

   return A*np.sin(2*np.pi*f*t)

# call and test your function here #
fs =44100   
N = 8000
freq = 20
amplitude = 4

sin=create_sine_wave(freq,amplitude,fs,N)
#plt.plot(sin)
#graph("Signal sinusoïdale de fréquence 20 Hz","Echantillon","Amplitude")
#plt.show()

# %%
from glob import glob
import scipy.io.wavfile as wf
import audiofile 


#!!!!!!!!!!!!!!!!!ATTENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 #!!! Ne pas appeler les variables signal car il peut confonde avec signal. de scipy!!!
 #J'ai donc remplacer tous les "signal" --> "signal_audio"
#!!!!!!!!!!!!!!!!Attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 


def read_wavefile(path):

    signal_audio, sampling_rate = audiofile.read(path)

    return signal_audio,sampling_rate

# call and test your function here #
LocateClaps = "LocateClaps"
files = glob(f"{LocateClaps}/*.wav")
x=len(files)
# Chemin vers le fichier spécifique
all_signals = []

# Lire et traiter chaque fichier
for file in files:
    signal_audio, sampling_rate = read_wavefile(file)
    all_signals.append(signal_audio)  # Ajouter le signal à la liste
#plt.plot(all_signals[0])
#graph({files[0]},"Temps","Amplitude")
#plt.show()
    
# Afficher le nombre total de fichiers traités
print(f"{len(files)} fichiers ont été traités.")

# %% [markdown]
# ### 1.2 Buffering
#g
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
my_buffer = create_ringbuffer(maxlen)
for i, sample in enumerate(signal_audio):
    my_buffer.append(sample)
    # your code here #

# %% [markdown]
# ### 1.3 Pre-processing
# #### 1.3.1 Normalisation
#a
# %%
def normalise(s):
    
    max_val=np.max(np.abs(s))
    if max_val == 0:
        return s

    return s/max_val

# call and test your function here #
sin_normalise = normalise(sin)

#plt.plot(sin, label="Original")
#plt.plot(sin_normalise, label="Normalisé")
#graph("Comparaison du signal original et normalisé","Echantillon","Amplitude")
#plt.legend()
#plt.show()
"""all_signals_normalise = []
for signal in all_signals:
    signal_normalise = normalise(signal)
    all_signals_normalise.append(signal_normalise)"""
# %% [markdown]
# #### 1.3.2 Downsampling
#g
# %%
## 1 - spectral analysis via spectrogram

'''specific_file = "LocateClaps\\M1_0.wav"#J'ai besoin signal audio 
signal_audio, sampling_rate = read_wavefile(specific_file)


plt.specgram(signal_audio, Fs=sampling_rate )
plt.title("Spectrogram")
plt.show()'''

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
#a


# wp = 16000 ws = 20000 gpass = 5db gstop = 40db
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellipord.html#scipy.signal.ellipord
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html
## 3 - Decimation
def downsampling(sig, B, A, M):

    signal_filtered = signal.lfilter(B,A,sig)

    signal_decimated = signal_filtered[::M]
     
    return signal_decimated


# call and test your function here
sin1=create_sine_wave(8500,1000,fs,200)
sin2=create_sine_wave(7500,20,fs,200)
'''M=3
sin_test = sin1 + sin2
B_cauer,A_cauer = create_filter_cauer(16000,20000,5,40,fs)
sin_test_downsampled = downsampling(sin_test,B_cauer,A_cauer,M)

w,h=signal.freqz(sin_test_downsampled,fs=fs)
plt.plot(w,abs(h))

plt.show()'''

# %% [markdown]
# ### 1.4 Cross-correlation
#g
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
    
    out = np.fft.fftshift(corr.real)

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
#a
# %%
def TDOA(xcorr):
    
    max_index = np.argmax(np.abs(xcorr))

    sample_offset = max_index - (len(xcorr) / 2)
    time_delay = sample_offset / (44100/3)
    print(sample_offset)
    return time_delay

# %% [markdown]
# #### 1.5.2 Equation system
#g
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
    angle=(np.arctan2(coordinates[1],coordinates[0]))
    angle=np.degrees(angle)
    if angle<0:
        angle+=360
    
    return angle

# Vérification des fichiers trouvés
if len(files) < 3:
    raise ValueError("Au moins trois fichiers audio sont nécessaires pour calculer les TDOA.")

# Lecture des fichiers audio
signals = []
sampling_rate = None

for file in files:
    signal_audio, fs = read_wavefile(file)
    signals.append(signal_audio)
    if sampling_rate is None:
        sampling_rate = fs
    elif sampling_rate != fs:
        raise ValueError("Tous les fichiers doivent avoir le même taux d'échantillonnage.")



# Calcul des TDOA à partir des signaux des microphones
xcorr_01 = fftxcorr(signals[0], signals[13])
tdoa_01 = TDOA(xcorr_01)

xcorr_02 = fftxcorr(signals[0], signals[24])
tdoa_02 = TDOA(xcorr_02)

# Affichage des résultats
print("TDOA entre micro 1 et micro 2 :", tdoa_01)
print("TDOA entre micro 1 et micro 3 :", tdoa_02)

# Vous pouvez ensuite utiliser les TDOA pour localiser la source sonore
# Exemple (supposant que la fonction localize_sound est définie) :
deltas = [tdoa_01, tdoa_02]
coordinates = localize_sound(deltas)
angle = source_angle(coordinates)

print("Coordonnées de la source sonore :", coordinates)
print("Angle par rapport à l'axe x :", angle)

