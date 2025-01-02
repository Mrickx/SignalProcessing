# %% [markdown]
# # Signal Processing Project: real-time sound localisation

# %% [markdown]
# ## 1 Offline system
# ### 1.1 Data generation and dataset
#ici

# %%
import numpy as np
import matplotlib.pyplot as plt
from Graph import graph
import os 

def create_sine_wave(f, A, fs, N):
    
   t=np.arange(0,N/fs,1/fs)
   

   return A*np.sin(2*np.pi*f*t)

# call and test your function here #
fs =44100   
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

    signal, sampling_rate = audiofile.read(path)

    return signal,sampling_rate

# call and test your function here #
LocateClaps = "LocateClaps"
files = glob(f"{LocateClaps}/*.wav")
x=len(files)
# Chemin vers le fichier spécifique
all_signals = []

# Lire et traiter chaque fichier
for file in files:
    signal, sampling_rate = read_wavefile(file)
    all_signals.append(signal)  # Ajouter le signal à la liste
plt.plot(all_signals[0])
graph({files[0]},"Temps","Amplitude")
plt.show()
    
# Afficher le nombre total de fichiers traités
print(f"{len(files)} fichiers ont été traités.")

def normalise(s):
    
    max_val=np.max(np.abs(s))
    if max_val == 0:
        return s

    return s/max_val

sin_normalise = normalise(sin)

plt.plot(sin, label="Original")
plt.plot(sin_normalise, label="Normalisé")
graph("Comparaison du signal original et normalisé","Echantillon","Amplitude")
plt.legend()
plt.show()

# call and test your function here #
"""all_signals_normalise = []
for signal in all_signals:
    signal_normalise = normalise(signal)
    all_signals_normalise.append(signal_normalise)"""

