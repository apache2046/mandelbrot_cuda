import numpy as np
import matplotlib.pyplot as plt
from ctypes import cdll, c_int, c_float, c_double
import ctypes
import cv2
from scipy.interpolate import PchipInterpolator
from numba import jit


x = [0.0, 0.16, 0.42, 0.6425, 0.8575]
y = [[  0,   7, 100],
     [ 32, 107, 230],
     [237, 255, 255],
     [255, 170,   0],
     [  0,   2,   0]]
interp = PchipInterpolator(x, y)
lut = np.zeros((256, 1, 3), dtype=np.uint8)
for i in range(256):
  c = interp(i/255)
  c[c<0] = 0 
  lut[i] = [c] 
print(lut.shape, lut)



# Load the CUDA shared library
cudalib = cdll.LoadLibrary('./mandelbrot.so')


# Set the image size and region of the complex plane to render
width = 7680 * 1
height = 4320 * 1
xMin = -2.0
xMax = 1.0
yMin = -1.5
yMax = 1.5
whr = width / height

maxIter = 2000

lut2 = np.zeros((maxIter+1, 3), dtype=np.uint8)
for i in range(maxIter+1):
  c = interp(i/(maxIter+1))
  c[c<0] = 0 
  lut2[i] = c 
print(lut2.shape, lut2)

cudalib.mandelbrot2.argtypes = [ctypes.POINTER(ctypes.c_int), c_int, c_int, c_double, c_double, c_double, c_double, c_int]
# Allocate memory for the output image

scale = 2.0
POI = [[0.001643721971153, -0.822467633298876],
       [-1.769383179195515018213847, 0.0042368479187367722149265],
       [0.281717921930775, 0.5771052841488505],
       [0.432539867562512, 0.226118675951818],
       [0.3602404434376143632361252, -0.641313061064803174860375]]
@jit
def lut_smooth(img, src, lut_s):
    for i in range(height):
        for j in range(width):
            img[i,j] = lut_s[src[i,j]]
for poi_idx, location in enumerate(POI):
    cx, cy = location
    scale = 2.0
    for idx in range(8000):
        scale *= 0.995
        print(idx, scale,  cx - scale, cx + scale, cy + scale, cy - scale, maxIter)
        output = np.empty((height, width), dtype=np.int32)
        
        # create a pointer to the numpy array data
        data_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        # Call the CUDA kernel to compute the Mandelbrot set
        #cudalib.mandelbrot_kernel(output, width, height, xMin, xMax, yMin, yMax)
        cudalib.mandelbrot2(data_pointer, width, height, (cx - scale*whr), (cx + scale*whr), cy + scale, cy - scale, maxIter)
        
        img3 = np.zeros((height, width, 3), dtype=np.uint8)
        lut_smooth(img3, output, lut2)
        img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        img3 = cv2.resize(img3, (width // 2, height // 2), interpolation = cv2.INTER_AREA) 
        cv2.imwrite(f"imgs.{poi_idx+1}/{idx:04}.png", img3)
        if output.sum() >= (maxIter-1) * width * height:
           print("black")
           break
