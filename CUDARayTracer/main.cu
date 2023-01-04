
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <iostream>

#include "utility.h"
#include "vec3.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "material.cuh"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "'\n";

        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 ray_color(const ray& r, hittable** world, curandState *local_rand_state) {
    ray curr_ray = r;
    vec3 curr_attenuation(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(curr_ray, 0.0001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(curr_ray, rec, attenuation, scattered, local_rand_state)) {
                curr_attenuation = curr_attenuation * attenuation;
                curr_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }

        }
        else {
            vec3 unit_direction = unit_vector(curr_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return curr_attenuation * c;
        }
    }
    return vec3(0.5, 0.0, 0.5); //has bounced 50 times

}

__global__ void render(vec3* fb, int max_x, int max_y, int samples, vec3 lower_left_corner,
    vec3 horizontal, vec3 vertical, vec3 origin, hittable** world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y))
        return;

    int pixel_idx = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_idx];
    vec3 col(0, 0, 0);
    for (int s = 0; s < samples; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        
        
        ray r(origin, lower_left_corner + u * horizontal + v * vertical);
        col += ray_color(r, world, &local_rand_state);
    }
    fb[pixel_idx] = col/float(samples);
}

__global__ void create_world(hittable** d_list, hittable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0, 0, -1), 0.5,
            new lambertian(vec3(0.1, 0.2, 0.5)));
        d_list[1] = new sphere(vec3(0, -100.5, -1), 100,
            new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1, 0, -1), 0.5,
            new metal(vec3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new sphere(vec3(-1, 0, -1), 0.5,
            new dielectric(1.5));
        d_list[4] = new sphere(vec3(-1, 0, -1), -0.45,
            new dielectric(1.5));
        *d_world = new hittable_list(d_list, 5);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

int main()
{
    // Image

    const int image_width = 1200;
    const int image_height = 600;
    int samples = 100;


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

    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    // setting up random values
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render <<<blocks, threads>>> (fb, image_width, image_height, samples,
                                  vec3(-2.0, -1.0, -1.0),
                                  vec3(4.0, 0.0, 0.0),
                                  vec3(0.0, 2.0, 0.0),
                                  vec3(0.0, 0.0, 0.0),
                                  d_world, d_rand_state);
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

            //gamma-correct for gamma=2.0 color has already been divided by number of samples
            float scale = 1.0;
            r = sqrt(scale * r);
            g = sqrt(scale * g);
            b = sqrt(scale * b);

            int ir = static_cast<int>(256 * clamp(r, 0.0, 0.999));
            int ig = static_cast<int>(256 * clamp(g, 0.0, 0.999));
            int ib = static_cast<int>(256 * clamp(b, 0.0, 0.999));

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


