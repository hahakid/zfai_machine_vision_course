import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../data/hand.png', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))  # magnitude spectrum

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

w, h = img.shape
ww = int(w / 2)
hh = int(h / 2)
fshift[ww-30:ww+30, hh-30:hh+30] = 0  # 中间过滤低频 (这里操作的不是像素坐标，而是频域) remove the low frequencies by masking
f_ishift = np.fft.ifftshift(fshift)  # inverse FFT
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(132)
plt.imshow(img_back, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("hight pass filtering")
plt.subplot(133)
plt.imshow(img_back)  # High Pass Filtering is an edge detection operation
plt.xticks([])
plt.yticks([])
plt.title("Jet")
plt.show()

# First channel will have the real part of the result and second channel will have the imaginary part of the result.
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

# low pass filter
mask = np.zeros((w, h, 2), np.uint8)
mask[ww-30:ww+30, hh-30:hh+30] = 1
fshift = dft_shift * mask

#dft_shift[ww-30:ww+30, hh-30:hh+30] = 0

f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(img_back, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])
plt.show()












