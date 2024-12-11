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
import re

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
# Fonction de clé de tri
def sorting_key(f):
    match = re.search(r'M(\d+)_(\d+)', f)
    if match:
        group = int(match.group(1))  # Numéro après 'M'
        angle = int(match.group(2))  # Angle
        return (group, angle)
    return (float('inf'), float('inf'))  # Pour gérer les fichiers sans correspondance

# Trier par groupe (M1, M2, etc.) et angle croissant
files = sorted(files, key=sorting_key)

print(files)
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
M=3
sin_test = sin1 + sin2
B_cauer,A_cauer = create_filter_cauer(8000,8500,0.1,70,44100)
'''sin_test_downsampled = downsampling(sin_test,B_cauer,A_cauer,M)

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
    
    corr = np.fft.fftshift(corr.real)

    out=corr

    return out
    
# call and test your function here #

xcorr_fftconv = sc.fftconvolve(sin1, sin2[::-1], 'full') # [::-1] flips the signal but you can also use np.flip()

xcorr = fftxcorr(sin1,sin2)

n_conv = np.arange(-(len(sin1) - 1), len(sin1))
n_fft = np.arange(-len(sin1) + 1, len(sin1))

'''plt.figure(figsize=(10, 4))
plt.plot(n_fft, xcorr, label='Autocorrelation (fftxcorr)', color='blue')
plt.plot(n_conv, xcorr_fftconv, label='Autocorrelation (fftconvolve)', color='orange', linestyle='dashed')
plt.title("Comparison of Autocorrelation Methods")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()'''

# %% [markdown]
# ### 1.5 Localisation
# #### 1.5.1 TDOA
#a
# %%
def TDOA(xcorr,fs=44100):
    
    max_index = np.argmax(np.abs(xcorr))
    sample_offset = max_index - (len(xcorr) / 2)
    time_delay=sample_offset/fs
    
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




# Génération de deux signaux sinusoïdaux avec un décalage temporel
fs = 44100  # Fréquence d'échantillonnage
freq = 1000  # Fréquence de la sinusoïde (Hz)
amplitude = 1  # Amplitude
duration = 0.1  # Durée du signal (secondes)
delay_in_seconds = 0.002  # Décalage temporel (secondes)
delay_in_samples = int(delay_in_seconds * fs)  # Décalage en échantillons

# Générer le premier signal sinusoïdal
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
sin1 = amplitude * np.sin(2 * np.pi * freq * t)

# Générer le second signal avec un décalage
sin2 = np.zeros_like(sin1)
sin2[delay_in_samples:] = sin1[:-delay_in_samples]



# Calcul de la corrélation croisée
xcorr = fftxcorr(sin1, sin2)

# Calcul du TDOA
time_delay = TDOA(xcorr, fs=fs)
time_delay = np.abs(time_delay)
print(f"Décalage temporel mesuré : {time_delay:.6f} s")
print(f"Décalage réel : {delay_in_seconds:.6f} s")

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
    


def accuracy(pred_angle, gt_angle, threshold):
    
    angle_accuracy = np.abs(pred_angle - gt_angle)

    angle_accuracy = np.minimum(angle_accuracy, 360 - angle_accuracy)

    return angle_accuracy <= threshold




# call and test your function here #

all_signals_post = ['']*len(files)
for i in range(len(files)):
    temp = normalise(all_signals[i])
    all_signals_post[i] = downsampling(temp,B_cauer,A_cauer,M)
    
threshold = 10
for k in range (12):
    xcorr12 = fftxcorr (all_signals_post[k],all_signals_post[k+12])
    xcorr13 = fftxcorr (all_signals_post[k],all_signals_post[k+24])
    deltas = np.array([TDOA(xcorr12),TDOA(xcorr13)])
    xy=localize_sound(deltas)
    pred_angle=source_angle(xy)
    print(pred_angle)
    
    ## 1.6.2
    '''possible_angle = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for angle in possible_angle:
        for f in files:
                        if f'_{angle}.' in f:
                            # Extraire l'angle réel du nom du fichier
                            gt_angle = angle
                            # Vérifier la précision
                            if accuracy(pred_angle, gt_angle, threshold):
                                print(f"Précision atteinte pour l'angle {gt_angle}° avec estimation {pred_angle}°")
                            else:
                                print(f"Précision non atteinte pour l'angle {gt_angle}° avec estimation {pred_angle}°")'''
                
    










'''#ici je faais le preprocessing de tous les claps qui sont dans le dossier
#c'est les étapes qui sont dans le grand cadre dans la fig1 du rapport

your_signal=['']*len(files)
#on declare le filtre
B,A=create_filter_cauer(16000,20000,0.1,70,44100)
for i in range(len(files)):
    #on normalise
    temp=normalise(all_signals[i])
    #on filtre
    #temp=signal.lfilter(B,A,temp)
    #on fait un downsample
    #normalement on devrait faire tous les 2.7 mais on peut pas pcq c'est pas un entier
    #en arondissant à 3 on perd un peu d'informations mais c'est vrmt pas grand chose
    #(2.7 pcq les frequences importantes du signal vont de 0 à 8kHz donc 44100/16KHz)
    your_signal[i]=downsampling(temp,B,A,M)
#%%

plt.figure(figsize=[5,5])
#ici 12 pcq on a 12 angles différents dans les claps
for k in range(12):
    #le meme angle mais avec un microphone diff se repete tous les 12 claps
    #ici entre microphone 1 et 2
    xcorr12=fftxcorr(your_signal[k],your_signal[k+12])
    #ici entre microphone 1 et 3
    xcorr13=fftxcorr(your_signal[k],your_signal[k+24])
    #on crée un vecteur avec les time delays
    #element 1 = time delay entre 1 et 2
    #element 2 = time delay entre 1 et 3
    deltas=np.array([TDOA(xcorr12),TDOA(xcorr13)])
    #ici ça resous les equations du rapport pour trouver la position X et Y de la source
    #xy[0] c'est la coordonnée X et xy[1] la coordonnée Y
    xy=localize_sound(deltas)
    #on fait une translation sur Y pour que ce soit bien centrée
    #jsp pq cette valeur là je l'ai prise apres avoir vu que les points sur le graph
    #à la fin étaient pas centrés faut que je demande au prof
    #pour déterminer les angles des sources
    angle=source_angle(xy)
    print(angle)
    
    #pour plot les positions des claps par rapport au rasberry (en 0,0)
    #si on veut voir les points 1 par 1 faut utiliser plt.figure()
    #plt.figure()
    plt.scatter(xy[0],xy[1])
    #pour dézoomer un peu ça permet de voir qu'on a pas un beau cercle
    plt.axis([-1.5, 1.5,-1.5,1.5 ])
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.grid()
    plt.show()
    #on voit que ça forme pas vrmt un cercle parfait pcq le downsampling nous
    #a fait perdre des informations
    #si on fait pas de downsampling on a un beau cercle'''
