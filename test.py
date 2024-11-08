from PIL import Image
import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt

im_gray = Image.open("USAF-1951.png").convert("L")
im_gray_512 = np.array(im_gray.crop((180, 190, 180+512, 190+512)))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(im_gray_512, cmap="gray")
f_im = ffp.fft2(im_gray_512)
shift_f_im = ffp.fftshift(f_im)
ax[1].imshow(np.log10(np.abs(shift_f_im)), cmap="gray")

fig.savefig("falcon512_spectrum.png")