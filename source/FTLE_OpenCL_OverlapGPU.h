/*
 *  FTLE_OpenCL_OverlapGPU.h
 *  FTLE2D
 *
 *  Created by Diego Rossinelli on 7/18/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <map>
#include <vector>
using namespace std;

#include "FTLE_OpenCL.h"
#include "EngineCL.h"
#include "ProfilerCL.h"

class FTLE_OpenCL_OverlapGPU : public FTLE_OpenCL
{
protected:
	
	cl::CommandQueue cmdq2;
	
	// Host resources
	cl::Image2D hostimgUm, hostimgVm;
	
	//additional resources for overlap
	cl::Image2D imgUnext, imgVnext;
	float * pinnedU, *pinnedV;
	cl::Event evWriteNextField2[2];
	
	
	void _flush();
	void _update_clprofiler();
	void _map_pinned();
	void _unmap_pinned();
	
	cl::Event _load_image(const float * const srcptr, float * const pinned, cl::Image2D& imgDest, const bool blocking, cl::CommandQueue commandqueue, const int numevents, ...);	
	
	//main helper method
	Real * _FTLE(queue<float *> u, vector<Real> dt, const int frame, const Real finalT);
	
public:
	
	FTLE_OpenCL_OverlapGPU(const int argc, const char ** argv, int sX, int sY, Real dh, bool bProfileOCL=true);	
};