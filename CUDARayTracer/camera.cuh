//#ifndef CAMERA_H
//#define CAMERA_H
//
//#include "utility.h"
////#include "vec3.cuh"
//
//class camera {
//public:
//	camera(
//		point3 lookfrom,
//		point3 lookat,
//		point3 vup,
//		float vfov,	// vertical fov in degrees
//		float aspect_ratio,
//		float aperture,
//		float focus_dist
//	) {
//
//		auto theta = deg_to_rad(vfov);
//		auto h = tan(theta / 2);
//		auto viewport_height = 2.0 * h;
//		auto viewport_width = viewport_height * aspect_ratio;
//
//		w = unit_vector(lookfrom - lookat);
//		u = unit_vector(cross(vup, w));
//		v = cross(w, u);
//		//auto focal_length = 1.0;
//
//		origin = lookfrom;
//		horizontal = focus_dist * viewport_width * u;
//		vertical = focus_dist * viewport_height * v;
//		lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
//		lens_radius = aperture / 2;
//	}
//
//	ray get_ray(float s, float t) const {
//		//vec3 rd = lens_radius * random_in_unit_disk();
//		vec3 offset = u * rd.x() + v * rd.y();
//
//		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset); //should be -offset at end
//
//	}
//
//private:
//	point3 origin;
//	point3 lower_left_corner;
//	vec3 horizontal;
//	vec3 vertical;
//	vec3 u, v, w;
//	float lens_radius;
//
//};
//
//#endif