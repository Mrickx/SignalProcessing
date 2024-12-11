import numpy as np
import matplotlib.pyplot as plt
from Graph import graph
import os 
from scipy import signal

def create_sine_wave(f, A, fs, N):
    
   t=np.arange(0,N/fs,1/fs)
   

   return A*np.sin(2*np.pi*f*t)

def read_wavefile(path):

    signal_audio, sampling_rate = audiofile.read(path)

    return signal_audio,sampling_rate

def create_ringbuffer(maxlen):
    
    # your code here #
    out = deque(maxlen=maxlen)

    return out

def normalise(s):
    
    max_val=np.max(np.abs(s))
    if max_val == 0:
        return s

    out = s/max_val 

    return out

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

def downsampling(sig, B, A, M):

    signal_filtered = signal.lfilter(B,A,sig)

    signal_decimated = signal_filtered[::M]
     
    return signal_decimated

def fftxcorr(in1, in2):
    
    # your code here #
    
    n = len(in1) + len(in2) - 1
    
    FFT1 = np.fft.fft(in1,n=n)
    FFT2 = np.fft.fft(in2,n=n)
    
    corr = np.fft.ifft(FFT1 * np.conj(FFT2))
    
    out = corr

    return out

def TDOA(xcorr):
    
    max_index = np.argmax(np.abs(xcorr))

    sample_offset = max_index - (len(xcorr) / 2)
    
    return sample_offset

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
    
    hypotenus = np.sqrt(x**2 + y**2)
    
    out = np.arcsin(x/hypotenus)

    return out

def accuracy(pred_angle, gt_angle, threshold):
    
    angle_accuracy = np.abs(pred_angle - gt_angle)

    angle_accuracy = np.min(angle_accuracy, 360 - angle_accuracy)

    return angle_accuracy <= threshold

def func_example(a, b):
    return a*b

def time_delay(func, args):
    start_time = time_ns()
    out = func(*args)
    end_time = time_ns()
    print(f"{func.__name__} in {end_time - start_time} ns")
    return out

def callback(in_data, frame_count, time_info, flag):
    global BUFFERS
    data = np.frombuffer(in_data, dtype=np.int16)
    BUFFERS[0].extend(data[0::RESPEAKER_CHANNELS])
    BUFFERS[1].extend(data[2::RESPEAKER_CHANNELS])
    BUFFERS[2].extend(data[4::RESPEAKER_CHANNELS])
    return (None, pyaudio.paContinue)

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