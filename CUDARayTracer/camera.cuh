#ifndef CAMERA_H
#define CAMERA_H

#include "utility.h"
#include "vec3.cuh"

class camera {
public:
	__device__ camera(
		point3 lookfrom,
		point3 lookat,
		point3 vup,
		float vfov,	// vertical fov in degrees
		float aspect_ratio,
		float aperture,
		float focus_dist
	) {

		float theta = deg_to_rad(vfov);
		float h = tan(theta / 2);
		float viewport_height = 2.0f * h;
		float viewport_width = viewport_height * aspect_ratio;

		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		float focal_length = 1.0f;

		origin = lookfrom;
		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
		lens_radius = aperture / 2;
	}

	__device__ ray get_ray(float s, float t, curandState* local_rand_state) const {
		vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
		vec3 offset = u * rd.x() + v * rd.y();

		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);

	}

private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;

};

#endif