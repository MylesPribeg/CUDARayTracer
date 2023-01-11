# CUDARayTracer
## About
This is the V1.0 Raytracer from the Raytracer repository translated to run in CUDA. 

## Performance
Below is a performance comparison between using the GPU and CPU. The CPU tests were run using the the V1.0 of the Raytracer (with the addition of a timer to accurately measure render time, and the multithreading code found in later revisions), while the GPU tests were run using the code in this repository. The tests were run on a Ryzen 7 2700 (16 cores) with 16 GB of RAM and a Nvidia 1080Ti.


| Device              | Time to Render | Speedup |
| ------------------- | -------------- | ------- |
| 1 Thread on CPU     | 69.007s        | 1.0x    |
| 10 Threads on CPU   | 12.579s        | 5.48x   |
| Using GPU (CUDA)    | 3.693s         | 18.69x  |
  

Render times are the average run time of 3 runs, and do not include set up time of the program (allocating memory, generating random numbers, etc.)

## Rendered Image
![alt text](https://github.com/MylesPribeg/CUDARayTracer/blob/master/RenderedImage.png)
