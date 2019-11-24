# btpython
#Bai tap nhom python

# bai 1:
import math
import numpy as np
import matplotlib.pyplot as plt

f = 60.0
T = 1 / f
fs = 1000
N = 5
A = 1
n = 999

t = np.arange(0, N * T , N * T / float(fs))

some_creepy_wave = np.zeros(fs)

for i in range(n):
    harmonic = 1 / float(2*i+1) / float(2*i+1) * np.sin(2 * math.pi * (2*i+1) * f * t)
    some_creepy_wave += harmonic

sin_t = A * np.sin(2 * math.pi * f * t)

plt.plot(t, sin_t, 'r', label='s1')
plt.savefig('s1.png')

plt.clf()

plt.plot(t, some_creepy_wave, 'b', label='sine wave')
plt.savefig('s2.png')

plt.plot(t, sin_t, 'r', label='s1')
plt.savefig('s1&s2.png')

# plt.show() 


# bai 2:

import math
import wave
import struct
import os

# 44100 is the industry standard sample rate - CD quality
sample_rate = 44100.044

def make_audio(duration_milliseconds=250, volume=0.5):
    audio = []
    freq_nor = 440
    freq_rate=math.pow(2, 1/12.0)
    num_samples_per_node = duration_milliseconds * (sample_rate / 1000.0)

    for i in range(12):
        freq = freq_nor*math.pow(freq_rate,i-8)

        for x in range(int(num_samples_per_node)):
            sample = volume * math.sin(2 * math.pi * freq * ( x / sample_rate ))
            audio.append(sample)

    return audio

def save_wav(audio, file_name):

    wav_file=wave.open(file_name,"w")

    # wav params
    nchannels = 1
    sampwidth = 2
    nframes = len(audio)
    comptype = "NONE"
    compname = "not compressed"
    wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))

    # WAV files here are using short, 16 bit, signed integers for the 
    # sample size.  So we multiply the floating point data we have by 32767, the
    # maximum value for a short integer.  NOTE: It is theortically possible to
    # use the floating point -1.0 to 1.0 data directly in a WAV file but not
    # obvious how to do that using the wave module in python.
    for sample in audio:
        wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))

    wav_file.close()

    return


audio = make_audio()
save_wav(audio, os.path.join('bai2',"output.wav"))

# bài 3:
import numpy as np
import matplotlib.pyplot as plt
# Đọc vào file ảnh image.png
img = plt.imread('image.png')

'''
Thực hiện yêu cầu chuyển đổi ảnh màu sang ảnh đen trắng
'''

# Hàm chuyển đổi rgb sang đen trắng
def rgb_to_gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Vẽ ảnh đen trắng và lưu vào file dentrang.png
fig1 = plt.figure()
plt.axis('off')
img = plt.imread('image.png')
gray = rgb_to_gray(img)
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()
fig1.savefig('dentrang.png')

'''
Thực hiện việc tách ảnh thành 3 kênh R, G, B
'''

# Thực hiện tạo red channel và lưu vào file redchannel.png
r = img.copy()
# set green and blue channels to 0
r[:, :, 1] = 0
r[:, :, 2] = 0
fig2 = plt.figure()
plt.axis('off')
plt.imshow(r)
fig2.savefig('redchannel.png')

# Thực hiện tạo green channel và lưu vào file greenchannel.png
g = img.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0
fig3 = plt.figure()
plt.axis('off')
plt.imshow(g)
fig3.savefig('greenchannel.png')

# Thực hiện tạo blue channel và lưu vào file bluechannel.png
b = img.copy()
# set blue and green channels to 0
b[:, :, 0] = 0
b[:, :, 1] = 0
fig4 = plt.figure()
plt.axis('off')
plt.imshow(b)
fig4.savefig('bluechannel.png')

'''
Thực hiện tổ hợp 3 ảnh R, G, B thành ảnh gốc
'''
red = plt.imread('redchannel.png')
green = plt.imread('greenchannel.png')
blue = plt.imread('bluechannel.png')
result = red.copy()
result[:, :, 1] = green[:, :, 1]
result[:, :, 2] = blue[:, :, 2]

# Thực hiện vẽ và lưu ảnh vào file rgb.png
fig5 = plt.figure()
plt.axis('off')
plt.imshow(result)
fig5.savefig('rgb.png')


#Bài 4:
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm

im = plt.imread('test.jpg').astype(float)

plt.figure()
plt.imshow(im, plt.cm.gray)
plt.title('Original image')
plt.savefig('gray_test.jpg')


im_fft = fftpack.fft2(im)
plt.figure()
plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
plt.colorbar()
plt.title('Fourier transform')
plt.savefig('fft_test.jpg')

im_new = fftpack.ifft2(im_fft).real
plt.figure()
plt.imshow(im_new, plt.cm.gray)
plt.title('Reconstructed Image')
plt.savefig('reconstruct_image.jpg')


# Bài 6:
import numpy as np
import scipy
from scipy import signal

K = 10
X = np.random.randint(K, size=8)
Y = np.random.randint(K, size=4)
# X = np.array([1,0,0,1,8,2])
# Y = np.array([1,0,1])
print('X = ', X)
print('Y = ', Y)

def linear_Conv(X, Y):
    print('linear convolution:')

    n = np.shape(X)[0]
    m = np.shape(Y)[0]

    # swap [y(0), y(1),...,y(m-1)] -> [y(m-1), y(m-2),...,y(0)]
    Y_swap = np.array([Y[m-1-i] for i in range(m)])

    # padding 0 to left and right of Y_swap( len(Y_pad) = 2*(n-1)+m)
    Y_pad = np.pad(Y_swap, (n-1,n-1), 'constant', constant_values=(0,0))
    #print('Y_pad = ', Y_pad)

    matrix_Y = np.array([Y_pad[(n-1)+m-i-1 : 2*(n-1)+m-i] for i in range(m+n-1)])
    #print('matrix_Y :\n', matrix_Y)

    matrix_X = X.T
    #print('matrix_X :', matrix_X)

    matrix_Z = np.dot(matrix_Y, matrix_X)
    print('matrix_Z : ', matrix_Z)

    Z = np.sum(matrix_Z, axis=0)
    return Z

def cyclic_Conv(X, Y):
    #print('cyclic convolution :')

    n = np.shape(X)[0]
    m = np.shape(Y)[0]

     # swap [y(0), y(1),...,y(m-1)] -> [y(m-1), y(m-2),...,y(0)]
    Y_swap = np.array([Y[m-1-i] for i in range(m)])

    # padding 0 to left of Y_swap (len(Y_pad = n))
    Y_pad = np.pad(Y_swap, (n-m,0), 'constant', constant_values=(0,0))
    Y_double = np.concatenate((Y_pad, Y_pad), axis=0)
    #print('Y_double = ', Y_double)

    matrix_X = X.T
    #print('matrix_X = ', matrix_X)

    matrix_Y = np.array([ Y_double[n-1-i : 2*n-1-i] for i in range(n)])
    #print('matrix_Y :\n ', matrix_Y)

    matrix_Z = np.dot(matrix_Y, matrix_X)
    print('matrix_Z = ', matrix_Z)

    Z = np.sum(matrix_Z, axis=0)
    return Z

print('linear convolution', linear_Conv(X, Y))    
print('cyclic convolution', cyclic_Conv(X, Y))
linear_convolve_csipy = signal.convolve(X, Y, mode='full', method='direct')
print('linear convolve csipy: ', linear_convolve_csipy) 
