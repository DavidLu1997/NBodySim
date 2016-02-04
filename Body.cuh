#ifndef BODY_CUH
#define BODY_CUH

#include "Point.cuh"

//Represents one circular body
class Body {
public:
	//Weight
	float weight;

	//Radius
	float radius;

	//Position
	Point position;

	//Acceleration
	Point acceleration;

	//Velocity
	Point velocity;

	//Constructor
	Body(Point l, float w = 1.0f, float r = 1.0f) {
		position = l;
		weight = w;
		radius = r;
	}
};

#endif