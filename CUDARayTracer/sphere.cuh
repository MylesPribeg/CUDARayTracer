#pragma once

#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.cuh"
#include "vec3.cuh"

class sphere : public hittable {
public:
	__device__ sphere() : radius(0) {}
	__device__ sphere(point3 cen, double r/*, shared_ptr<material> m*/) : center(cen), radius(r)/*, mat_ptr(m)*/ {};

	__device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& t) const override;

public:
	point3 center;
	double radius;
	//shared_ptr<material> mat_ptr;
};

__device__ bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {

	vec3 oc = r.origin() - center; //sphere form is: t^2b (dot) b + 2tb (dot) (A - C) + (A - C) (dot)(A - C) - r^2 = 0
	//since A, C, b, and r are known, it becomes a quadratic eqn with t unknown

	auto a = r.direction().length_squared();//vector dotted with itself is its length^2
	auto half_b = dot(oc, r.direction());	//some simplification of quadratic formula since b has a factor of 2
	auto c = oc.length_squared() - radius * radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant < 0) {
		return false;
	}
	auto sqrtd = sqrt(discriminant);

	//Find nearest root that lies within acceptable range
	auto root = (-half_b - sqrtd) / a;
	if (root <  t_min || root > t_max) {
		root = (-half_b + sqrtd) / a;
		if (root < t_min || root > t_max)
			return false;
	}

	rec.t = root;
	rec.p = r.at(rec.t); //point of intersection
	vec3 outward_normal = (rec.p - center) / radius;//returning a unit vector 
	rec.set_face_normal(r, outward_normal);
	//rec.mat_ptr = mat_ptr;// telling what type of material was hit

	return true;
}


#endif