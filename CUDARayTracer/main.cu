
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "utility.h"
#include "vec3.cuh"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "'\n";

        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 color(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(float* fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y))
        return;

    int pixel_idx = j * max_x * 3 + i * 3;
    fb[pixel_idx + 0] = float(i) / max_x;
    fb[pixel_idx + 1] = float(j) / max_y;
    fb[pixel_idx + 2] = 0.2;
}

int main()
{
    // Image

    const int image_width = 256;
    const int image_height = 256;


    int num_pixels = image_height * image_width;
    size_t fb_size = num_pixels * sizeof(vec3);
    
    //allocate framebuffer
    float* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // Render

    int tx = 8;
    int ty = 8;

    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads>>> (fb, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output FB as image
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            size_t pixel_index = j * 3 * image_height + i * 3;
            auto r = fb[pixel_index + 0];
            auto g = fb[pixel_index + 1];
            auto b = fb[pixel_index + 2];

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    checkCudaErrors(cudaFree(fb));
    return 0;
}


