
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "utility.h"
#include "vec3.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "'\n";

        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 ray_color(const ray& r, hittable** world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
        //return 0.5f * vec3(1.0, 0.5, 0.7);

    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }

}

__global__ void render(vec3* fb, int max_x, int max_y, vec3 lower_left_corner,
    vec3 horizontal, vec3 vertical, vec3 origin, hittable** world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y))
        return;

    int pixel_idx = j * max_x + i;

    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    fb[pixel_idx] = ray_color(r, world);
}

__global__ void create_world(hittable** d_list, hittable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new sphere(vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_list, 2);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
}

int main()
{
    // Image

    const int image_width = 1200;
    const int image_height = 600;


    int num_pixels = image_height * image_width;
    size_t fb_size = num_pixels * sizeof(vec3);


    //World
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    create_world<<<1, 1 >>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    //allocate framebuffer
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
    
    // Render

    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads>>> (fb, image_width, image_height,
                                  vec3(-2.0, -1.0, -1.0),
                                  vec3(4.0, 0.0, 0.0),
                                  vec3(0.0, 2.0, 0.0),
                                  vec3(0.0, 0.0, 0.0),
                                  d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output FB as image
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i ;
            auto r = fb[pixel_index][0];
            auto g = fb[pixel_index][1];
            auto b = fb[pixel_index][2];

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    return 0;
}


