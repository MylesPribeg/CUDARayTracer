#pragma once

#ifndef MATERIAL_H
#define MATERIAL_H

#include "utility.h"
#include "hittable.cuh" //added by me

//struct hit_record;

class material {
public:
	__device__ virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
		curandState* local_rand_state
	) const = 0;

};

class lambertian : public material {
public:
	__device__ lambertian(const color& a) : albedo(a) {}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
		ray& scattered, curandState* local_rand_state) const override {
		auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);

		//catch scatter directions near 0 (if unit vec is opposite of normal)
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal;

		scattered = ray(rec.p, scatter_direction);
		attenuation = albedo;
		return true;
	}

public:
	color albedo;
};


class metal : public material {
public:
	__device__ metal(const color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation,
		ray& scattered, curandState* local_rand_state) const override {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);//if ray is facing outwards return true
	}

public:
	color albedo;
	float fuzz;
};

class dielectric : public material {
public:
	__device__ dielectric(float ior) : ir(ior) {} //ior - index of refaction

	__device__ virtual bool scatter(
			const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, 
			curandState* local_rand_state
		) const override {

		attenuation = color(1, 1, 1);
		float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;//if not hitting from the front then IOR ratio is reciprocal

		vec3 unit_direction = unit_vector(r_in.direction());
		//vec3 refracted = refract(unit_direction, rec.normal, refraction_ratio);

		float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);//from dot product, cannot be greater than 1
		float sin_theta = sqrtf(1 - cos_theta * cos_theta);//from trig equality sin = sqrt(1 - cos^2(theta))

		bool cannot_refract = refraction_ratio * sin_theta > 1.0f; //for Total Internal Reflection
		vec3 direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(local_rand_state))
			direction = reflect(unit_direction, rec.normal);
		else
			direction = refract(unit_direction, rec.normal, refraction_ratio);

		scattered = ray(rec.p, direction);
		return true;
	}
public:
	float ir;

private:
	__device__ static float reflectance(float cosine, float ref_idx) {//for Fresnel
		//Schlick's approximation for reflectance
		float r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * powf(1 - cosine, 5);
	}

};

#endif // !MATERIAL_H
