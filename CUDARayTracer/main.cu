
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <iostream>
#include <time.h>

#include "utility.h"
#include "vec3.cuh"
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "camera.cuh"
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
        if ((*world)->hit(curr_ray, 0.001f, FLT_MAX, rec)) { //if lower limit is less than 0.001 it causes banding on large objects
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
    return vec3(0.0, 0.0, 0.0); //has exceeded max number of bounces

}

__global__ void render(vec3* fb, int max_x, int max_y, int samples,
                       camera** cam, hittable** world, curandState *rand_state) {
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
        
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += ray_color(r, world, &local_rand_state);
    }
    fb[pixel_idx] = col/float(samples);
}

#define RND (curand_uniform(&local_rand_state))
__global__ void create_world(hittable** d_list, hittable** d_world,
    camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        //d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
        //    new lambertian(vec3(0.5, 0.5, 0.5)));
        d_list[0] = new sphere(vec3(0, -1500, -1), 1500,
            new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[1] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[2] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[3] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        int i = 4;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new lambertian(vec3(RND, RND, RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }

        *rand_state = local_rand_state;
        *d_world = new hittable_list(d_list, 22*22+4);


        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;// (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
    delete* d_camera;
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
    int samples = 10;


    int num_pixels = image_height * image_width;
    size_t fb_size = num_pixels * sizeof(vec3);

    //Performance 
    int tx = 8;
    int ty = 8;

    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);

    // setting up random values
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    render_init << <blocks, threads >> > (image_width, image_height, d_rand_state);


    //World
    hittable** d_list;
    int num_objects = 22 * 22 + 4;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_objects * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera)));
    create_world<<<1, 1 >>>(d_list, d_world, d_camera, image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    //allocate framebuffer
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
    
    // Render

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    clock_t start, stop;
    start = clock();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render <<<blocks, threads>>> (fb, image_width, image_height, samples,
                                  d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";


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
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    while (true);

    return 0;
}

