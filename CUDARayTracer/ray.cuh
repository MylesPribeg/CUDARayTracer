#pragma once

#include "vec3.cuh"

class ray {
public:
	__device__ ray() {} //default constructor
	__device__ ray(const point3& origin, const vec3& direction) { //point 3 is alias of vec3
		orig = origin;
		dir = direction;
	}

	__device__ point3 origin() const { return orig; }
	__device__ vec3 direction() const { return dir; }

	__device__ point3 at(float t) const { 
		return orig + t * dir;
	}

public:
	point3 orig;
	vec3 dir;

};
