#pragma once

#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.cuh"
#include "utility.h"

class material;

struct hit_record {
	point3 p;
	vec3 normal;
	//shared_ptr<material> mat_ptr;
	double t;
	bool front_face; //normal is always against direction of ray

	__device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0; //comparision evaluates to true or false
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {//this is an abstract class
public:
	__device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};

#endif