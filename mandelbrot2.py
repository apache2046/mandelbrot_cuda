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
cudalib.mandelbrot3.argtypes = [ctypes.POINTER(ctypes.c_int), c_int, c_int, c_double, c_double, c_double, c_double, c_double, c_int]
# Allocate memory for the output image

scale = 2.0
POI = [
       [ -1.769383179195515, -1.8213847286085474e-17, 0.004236847918736772, 2.14926507171368e-19 ],
       [ 0.36024044343761436, 3.2361252444495455e-18, -0.6413130610648031, -7.48603750151793e-17]
      ]
@jit
def lut_smooth(img, src, lut_s):
    for i in range(height):
        for j in range(width):
            img[i,j] = lut_s[src[i,j]]
for poi_idx, location in enumerate(POI):
    cx_h, cx_t, cy_h, cy_t = location
    scale = 2.0
    for idx in range(10000):
        scale *= 0.995
        print(idx, scale,  cx - scale, cx + scale, cy + scale, cy - scale, maxIter)
        output = np.empty((height, width), dtype=np.int32)
        
        # create a pointer to the numpy array data
        data_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        # Call the CUDA kernel to compute the Mandelbrot set
        #cudalib.mandelbrot_kernel(output, width, height, xMin, xMax, yMin, yMax)
        #cudalib.mandelbrot2(data_pointer, width, height, (cx - scale*whr), (cx + scale*whr), cy + scale, cy - scale, maxIter)
        cudalib.mandelbrot3(data_pointer, width, height, cx_h, cx_t, cy_h, cy_t, scale, maxIter)
        
        #output=(output / maxIter * 255).astype(np.uint8)
        #img2 = cv2.applyColorMap(output, cv2.COLORMAP_JET)
        #cv2.imwrite("2.png", img2)
        
        #img3 = np.broadcast_to(output.reshape(height,width,1), (height, width, 3))
        #print("\n\n", img3.shape, img3.dtype, lut.shape, lut.dtype)
        
        #img3 = cv2.LUT(img3, lut)
        #img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        #img3 = cv2.applyColorMap(output, cv2.COLORMAP_JET)
        img3 = np.zeros((height, width, 3), dtype=np.uint8)
        lut_smooth(img3, output, lut2)
        img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        img3 = cv2.resize(img3, (width // 2, height // 2), interpolation = cv2.INTER_AREA) 
        cv2.imwrite(f"imgs_new.{poi_idx}/{idx:05}.png", img3)
        if output.sum() >= (maxIter-1) * width * height:
           print("black")
           break
