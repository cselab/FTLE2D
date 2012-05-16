/*
 *  FTLE2D.h
 *  FTLE2D
 *
 *  Created by Christian Conti on 1/20/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

using namespace std;

#pragma once

#ifndef _DOUBLE
typedef float Real;
#else
typedef double Real;
#endif

#include <vector>
#include <queue>

#include "ArgumentParser.h"
#include "Profiler.h"
#include "OperatorM2P.h"

#include <xmmintrin.h>
#include <emmintrin.h>

class FTLE2D
{
protected:
	bool bVerbose;
	bool bProfile;

	int sizeX, sizeY, size;
	Real * particlePos;
	Real * particlePosStar;
	Real * pos0;
	Real * particleVel;
	Real * meshVel;
	Real * FTLEfield;
	
	vector<Real *> flowMapsX, flowMapsY;
	vector<Real *> collapsedMapsX, collapsedMapsY; 
	
	Real h,invh;
	OperatorM2P * m2p;
	
	vector<Real *> vFTLEFields;
	
	ArgumentParser parser;
	Profiler profiler;
	
	// Options
	bool bCentered;
	bool bGFLOPs;
	string sFilePattern;
	//int nFrames;
		

	// Unsteady Flow Methods
	virtual Real* _FTLE(queue<Real *> u, vector<Real> dt, const int frame, const Real finalT);
	void _InitParticlesPos();
	void _AdvectionSubstep2(Real dt, const int channel);
	void _AdvectionSubstep3(Real dt, const int channel);
	void _TrustedAdvection(queue<Real *>& u, vector<Real>& dt, const int frame, const Real diffT);
	virtual void _FillMeshVel(queue<float *>& u);
	virtual void _FillFieldFromField(const float * const u, float * const output);
	Real _FindMaxDt(Real * u);
	
	// Common Methods
	void _ComputeFTLE(const Real T);
	Real _ComputeEigenvalue(int x, int y);
	virtual void _Euler2(Real dt, int ic);
	virtual void _Euler3(Real dt, int ic);
	
public:
	FTLE2D(const int argc, const char ** argv, int sX, int sY, Real dh, bool bVerbose=true, bool bProfile=true);
	~FTLE2D();
	void clear();
	
	// FTLE for unsteady velocity field
	virtual vector<Real *> operator()(vector<Real *>& u, vector<Real>& dt, const Real T);
};
