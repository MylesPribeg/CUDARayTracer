#pragma once

#include <cmath>
#include <limits>
#include <memory>
//#include <random>
#include <cstdlib>

using std::shared_ptr;
using std::make_shared;
using std::sqrt;


const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

//Utility Functions

inline float deg_to_rad(float degrees) {
	return degrees * pi / 180.0;
}

/*inline double random_double() {
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

inline double random_double(double min, double max) {
	static std::uniform_real_distribution<double> distribution(min, max);
	static std::mt19937 generator;
	return distribution(generator);
}*/

inline float random_float() {
	return rand() / (RAND_MAX + 1.0);
}

inline float random_float(float min, float max) {
	return min + (max - min) * random_float();
}

inline float clamp(float x, float min, float max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

//Common Headers

#include "ray.cuh"
#include "vec3.cuh"

