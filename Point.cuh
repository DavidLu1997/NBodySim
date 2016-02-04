#ifndef POINT_CUH
#define POINT_CUH

#include <cmath>

class Point {
public:
	float x, y, z;

	Point(float xi, float yi, float zi) {
		x = xi;
		y = yi;
		z = zi;
	}

	Point() {
		x = 0;
		y = 0;
		z = 0;
	}

	float magnitude() {
		return sqrt(x * x + y * y + z * z);
	}
};

#endif