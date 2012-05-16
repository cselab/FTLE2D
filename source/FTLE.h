/*
 *  FTLElib.h
 *  FTLElib
 *
 *  Created by Christian Conti on 1/20/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#include <vector>

using namespace std;

#ifndef _DOUBLE
typedef float Real;
#else
typedef double Real;
#endif

extern vector<Real *> libCore(const int argc, const char ** argv, int sX, int sY, Real dh, vector<Real *>& u, vector<Real>& dt, const Real T);

class FTLE2Dlib
{
	int argc;
	const char ** argv;
	int sX, sY;
	Real dh;
	
public:
	FTLE2Dlib(const int argc, const char ** argv, int sX, int sY, Real dh) : argc(argc), argv(argv), sX(sX), sY(sY), dh(dh)
	{
	}
	
	vector<Real *> operator()(vector<Real *>& u, vector<Real>& dt, const Real T)
	{
		return libCore(argc, argv, sX, sY, dh, u, dt, T);
	}
};