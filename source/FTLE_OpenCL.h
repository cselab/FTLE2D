/*
 *  FTLE_OpenCL.h
 *  FTLE2D
 *
 *  Created by Christian Conti on 1/20/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <map>
#include <vector>
#include <cstdarg>
using namespace std;

#include "FTLE2D.h"
#include "EngineCL.h"
#include "ProfilerCL.h"

class FTLE_OpenCL : public FTLE2D
{
protected:
	
	const bool bProfileOCL;
	const bool bWrapOnHost;
	
	cl::CommandQueue cmdq;

	EngineCL& engineCL;
	ProfilerCL trajectories_prof, prof;	
	cl::size_t<3> origin, region;
	size_t wgs_m2p, gls_m2p, wgs_euler, gls_euler, wgs_ftle, gls_ftle;
			
	// kernel and events
	cl::Kernel kerM2P, kerEu2, kerEu3, kerFTLE, kerInitXp, kerInitYp;
	cl::Event evWriteNextField[2];
	cl::Event evM2Pa[2], evM2Pb[2], evEuler2[2], evEuler3[2], evFTLE, evReadFTLE;
	
	// Context resources
	cl::Buffer bufX, bufY, bufXstar, bufYstar;
	cl::Buffer bufUp, bufVp;
	cl::Buffer bufFTLE;
	cl::Image2D imgUm, imgVm;
	
	void _initialize();
	void _lets_wait(const int nevents, ...);
	cl::Event _load_image(const float * const srcptr, cl::Image2D& imgDest, const bool blocking, cl::CommandQueue commandqueue, const int numevents, ...);	
	cl::Event _load_image(const float * const srcptr, cl::Image2D& imgDest, const bool blocking, cl::CommandQueue commandqueue, vector<cl::Event>);	

	cl::Event _initpos(cl::Kernel kerInit, cl::CommandQueue commandqueue, cl::Buffer bufResult) ;
	cl::Event _initpos(cl::Kernel kerInit, cl::CommandQueue commandqueue, cl::Buffer bufResult, vector<cl::Event> wait);
	
	cl::Event _eu(cl::Kernel kerEu, cl::CommandQueue commandqueue, cl::Buffer bufX, cl::Buffer bufU, cl::Buffer bufXstar, const float dt, const int nevents, ...) const;
	cl::Event _m2p(cl::CommandQueue commandqueue, cl::Image2D imgField, cl::Buffer bufX, cl::Buffer bufY, cl::Buffer bufResult, const int nevents, ...) ;
	
	void _compute_ftle(float * const destptr,  const Real finalT, const int nevents, ...);
	void _clprofile_update();
	void _clprofile_summary();
	
	//main helper method
	Real * _FTLE(queue<float *> u, vector<Real> dt, const int frame, const Real finalT);

	FTLE_OpenCL(const int argc, const char ** argv, int sX, int sY, Real dh, EngineCL& engineCL, bool bProfileOCL, bool bWrapOnHost);

public:
	
	FTLE_OpenCL(const int argc, const char ** argv, int sX, int sY, Real dh, bool bProfileOCL=true, bool bWrapOnHost=true);

	vector<Real *> operator()(vector<Real *>& u, vector<Real>& dt, const Real T);
};
