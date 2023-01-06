#pragma once //for VS
#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

using std::sqrt;

class vec3 {

public:
	__host__ __device__ vec3() : e{ 0,0,0 } {}//default constructor
	__host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}//non-default constructor
	//getters
	__host__ __device__ float x() const {return e[0];}
	__host__ __device__ float y() const {return e[1];}
	__host__ __device__ float z() const {return e[2];}

	//operators
	__host__ __device__ vec3 operator-() const {
		return vec3(-e[0], -e[1], -e[2]);
	}
	

	//There are two variants of this operator so that the elements returned can be changed
	__host__ __device__ float operator[](int i) const { return e[i]; }
	__host__ __device__ float& operator[](int i) { return e[i]; }

	__host__ __device__ vec3& operator+=(const vec3& v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];

		return *this;
	}

	__host__ __device__ vec3& operator*=(const float t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;

		return *this;
	}

	__host__ __device__ vec3& operator/=(const float t) {
		return *this *= 1 / t;
	}
	
	//other functions

	__host__ __device__ float length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	__host__ __device__ float length() const {
		return sqrtf(length_squared());

	}

	//__host__ __device__ inline static vec3 random() {
	//	return vec3(random_float(), random_float(), random_float());
	//}

	//__host__ __device__ inline static vec3 random(float min, float max) {
	//	return vec3(random_float(min, max), random_float(min, max), random_float(min, max));
	//}

	__host__ __device__ bool near_zero() const {
		//Return true if the vector is close to zero in all dimensions
		const float s = 1e-8;
		return (fabsf(e[0]) < s) && (fabsf(e[1]) < s) && (fabsf(e[2]) < s); //fabs = absolute value
	}

	float e[3];

};

//Type aliases for vec3

using point3 = vec3; //3D point
using color = vec3;  //RGB colour



//more utility functions/operator overloads

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
	//TRY THIS WITH JUST u[0] * v[0], etc. it should work because that operator is defined
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
	return t * v; //already defined
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t) {
	return (1 / t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
	return u[0] * v[0] 
		+ u[1] * v[1] 
		+ u[2] * v[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, vec3& v) {
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
				u.e[2] * v.e[0] - u.e[0] * v.e[2],
				u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

__device__ point3 random_in_unit_sphere(curandState* local_rand_state) {
	vec3 p;
	while (1) {
		p = 2 * vec3(curand_uniform(local_rand_state),
					 curand_uniform(local_rand_state),
					 curand_uniform(local_rand_state)) - vec3(1,1,1);

		if (p.length_squared() >= 1.0f) continue;
		return p;
	}
}

__device__ inline vec3 random_unit_vector(curandState* local_rand_state) {
	return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
	while (true) {
		vec3 p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

__host__ __device__ vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n) * n;
}

__host__ __device__ vec3 refract(const vec3& uv, const vec3& n, float rior) {//ratio of index of refractions "rior"
	auto cos_theta = fminf(dot(-uv, n), 1.0f);
	vec3 r_out_perp = rior * (uv + cos_theta * n);
	vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}

//__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
//	vec3 uv = unit_vector(v);
//	float dt = dot(uv, n);
//	float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
//	if (discriminant > 0) {
//		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
//		return true;
//	}
//	else
//		return false;
//}

#endif