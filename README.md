# DLSS In 3 Weekends

## Background

Real time raytracing has become a killer feature of many AAA game releases. With the advancement in deep-learning architectures, there has been a push to use machine learning to help accelerate the raytracing process in a variety of methods from sample prediction to image space denoising. One of the proposed solutions to this problem is to perform ray-tracing at a low resolution and low sample count, apply some learned upsampling technique to reach the desired resolution, and then apply a denoising filter to clean up the noise from the stochastic samples.

This is the approach essentially touted by Nvidia's DLSS system (CITATION HERE). For this project we seek to create a system that can similarly perform upsampling and denoising for a ray-traced image for use in real time applications. 

To achieve the goal of real-time performance, our upscaling and denoising system must meet a certain frame-time constraint. To make our lives a bit easier, we define a "real time"  performance to be a system that can go from a ray-trace call to final output in at least 33 ms (30 frames per second). However, we also want our model to produce somewhat accurate results, so we also constrain our model to have at least as good quality results as a full resolution 4spp ray traced image (which can be done in about 26 ms) in, at most, the same amount of time. 

What this means concretely is that our model must have a PSNR greater than or equal to a 4spp ray-traced image in our validation dataset.


## Approach
## Results



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTgyMzg1MjUzNSw4MzIyMTE2NywtMTQ2ND
U2OTAwNV19
-->