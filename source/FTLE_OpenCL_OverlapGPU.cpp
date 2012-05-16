/*
 *  FTLE_OpenCL_OverlapGPU.cpp
 *  FTLE2D
 *
 *  Created by Diego Rossinelli on 7/18/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */
#include <cassert>
#include <iomanip>
#include <sstream>

using namespace std;

#include "FTLE_OpenCL_OverlapGPU.h"

FTLE_OpenCL_OverlapGPU::FTLE_OpenCL_OverlapGPU(const int argc, const char ** argv, int sX, int sY, Real dh, bool bProfileOCL) : 
FTLE_OpenCL(argc, argv, sX, sY, dh, EngineCL::getInstance(2, bProfileOCL), bProfileOCL, false)
{	
	const int device_rank = parser("-device").asInt(0);
	cmdq2 = engineCL.getCommandQueue(1, device_rank);

	try
	{
		const cl::ImageFormat format(CL_RGBA,CL_FLOAT);
		const cl::Context ctext = engineCL.getContext();
		
		imgUnext = cl::Image2D(ctext, CL_MEM_READ_ONLY | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR , format, sizeX/4, sizeY);
		imgVnext = cl::Image2D(ctext, CL_MEM_READ_ONLY | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, format, sizeX/4, sizeY);
		
		hostimgUm = cl::Image2D(ctext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, format, sizeX/4, sizeY);
		hostimgVm = cl::Image2D(ctext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, format, sizeX/4, sizeY);
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
}

void FTLE_OpenCL_OverlapGPU::_map_pinned()
{
	try 
	{
		size_t pitch = 0;
		
		pinnedU = (float *)cmdq.enqueueMapImage(hostimgUm, CL_TRUE, CL_MAP_WRITE, origin, region, &pitch, NULL);
		pinnedV = (float *)cmdq2.enqueueMapImage(hostimgVm, CL_TRUE, CL_MAP_WRITE, origin, region, &pitch, NULL);
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
}

void FTLE_OpenCL_OverlapGPU::_unmap_pinned()
{
	try 
	{		
		cmdq.enqueueUnmapMemObject(hostimgUm, pinnedU);
		cmdq2.enqueueUnmapMemObject(hostimgVm, pinnedV);
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
}

cl::Event FTLE_OpenCL_OverlapGPU::_load_image(const float * const srcptr, float * const pinned, cl::Image2D& imgDest, const bool blocking, cl::CommandQueue commandqueue, const int numevents, ...)
{
	_FillFieldFromField(srcptr, pinned);

	vector<cl::Event> events;
	
	{
		va_list ev_list;
		
		va_start(ev_list, numevents);
		for(int i=0; i<numevents; i++)
			events.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}

	return FTLE_OpenCL::_load_image(pinned, imgDest, blocking, commandqueue, events);
}

void FTLE_OpenCL_OverlapGPU::_flush()
{
	cmdq.flush();
	cmdq2.flush();
}

void FTLE_OpenCL_OverlapGPU::_update_clprofiler()
{
	if (bProfileOCL)
	{
		_lets_wait(2, &evEuler3[0], &evEuler3[1]);
	
		trajectories_prof.update();
	}
}

Real * FTLE_OpenCL_OverlapGPU::_FTLE(queue<Real *> u_, vector<Real> dts, const int frame, const Real finalT)
{	
	if (u_.size() != dts.size() + 1)
	{
		if (bVerbose) cout << "Warning! The condition #dts = u.size() - 1 is not satisfied!" << endl << "#dts = " << dts.size() << ", u.size() = " << u_.size() << endl;
		if (bVerbose) cout << "We impose #dts := u.size() - 1" << endl << endl;
	}
		
	vector<Real *> u, v;

	const int N = u_.size();		
	for(int i=0; i<N; i++)	
	{
		Real * p = u_.front();
		u.push_back(p);
		v.push_back(p + sizeX*sizeY);
		
		u_.pop();
	}
	assert( u.size() == N );
	
	cl::Event &initUm = evWriteNextField[0], &initVm = evWriteNextField[1];
	cl::Event &initUnext = evWriteNextField2[0], &initVnext = evWriteNextField2[1];
	
	//first iteration
	{
		_map_pinned();

		cl::Event initXp = _initpos(kerInitXp, cmdq, bufX);
		cl::Event initYp = _initpos(kerInitYp, cmdq, bufY);		
		
		initUm = _load_image(u[0], pinnedU, imgUm, false, cmdq, 0);
		initVm = _load_image(v[0], pinnedV, imgVm, false, cmdq2, 0);
		initUnext = _load_image(u[1], pinnedU, imgUnext, false, cmdq2, 1, &initUm);
		
		_flush();

		initVnext = _load_image(v[1], pinnedV, imgVnext, false, cmdq2, 0);		
		evM2Pa[0] = _m2p(cmdq, imgUm, bufX, bufY, bufUp, 3, &initXp, &initYp, &initUm);
		evM2Pa[1] = _m2p(cmdq, imgVm, bufX, bufY, bufVp, 3, &initXp, &initYp, &initVm);

		_flush();

		evEuler2[0] = _eu(kerEu2, cmdq, bufX, bufUp, bufXstar, dts.front(), 1, &evM2Pa[0]);
		evEuler2[1] = _eu(kerEu2, cmdq2, bufY, bufVp, bufYstar, dts.front(), 1, &evM2Pa[1]);		
	}
	
	
	if (bProfile) profiler.push_start("OpenCL Advection Loop");
	//main iterations
	for(int i=1; i<N-1; i++)
	{
		_flush();

		swap(initUm, initUnext);
		swap(initVm, initVnext);
		swap(imgUm, imgUnext);
		swap(imgVm, imgVnext);
	
		initUnext = _load_image(u[i+1], pinnedU, imgUnext, false, cmdq, 1, &evM2Pa[0]);
		evM2Pb[0] = _m2p(cmdq2, imgUm, bufXstar, bufYstar, bufUp, 3, &evEuler2[0], &evEuler2[1], &initUm);
		evM2Pb[1] = _m2p(cmdq2, imgVm, bufXstar, bufYstar, bufVp, 3, &evEuler2[0], &evEuler2[1], &initVm);
	
		//_flush();

		evEuler3[0] = _eu(kerEu3, cmdq, bufX, bufUp, bufXstar, dts[i-1], 1, &evM2Pb[0]);
		evEuler3[1] = _eu(kerEu3, cmdq2, bufY, bufVp, bufYstar, dts[i-1], 1, &evM2Pb[1]);
		
		_flush();
		_update_clprofiler();
		
		initVnext = _load_image(v[i+1], pinnedV, imgVnext, false, cmdq2, 1, &evM2Pa[1]);
		evM2Pa[0] = _m2p(cmdq, imgUm, bufX, bufY, bufUp, 2, &evEuler3[0], &evEuler3[1]);
		evM2Pa[1] = _m2p(cmdq, imgVm, bufX, bufY, bufVp, 2, &evEuler3[0], &evEuler3[1]);
	
		//_flush();

		evEuler2[0] = _eu(kerEu2, cmdq, bufX, bufUp, bufXstar, dts[i], 1, &evM2Pa[0]);
		evEuler2[1] = _eu(kerEu2, cmdq2, bufY, bufVp, bufYstar, dts[i], 1, &evM2Pa[1]);
	}
	
	//last iteration
	{
		evM2Pb[0] = _m2p(cmdq, imgUnext, bufXstar, bufYstar, bufUp, 3, &evEuler2[0], &evEuler2[1], &initUnext);
		evM2Pb[1] = _m2p(cmdq2, imgVnext, bufXstar, bufYstar, bufVp, 3, &evEuler2[0], &evEuler2[1], &initVnext);
		evEuler3[0] = _eu(kerEu3, cmdq, bufX, bufUp, bufXstar, dts[N-2], 1, &evM2Pb[0]);
		evEuler3[1] = _eu(kerEu3, cmdq2, bufY, bufVp, bufYstar, dts[N-2], 1, &evM2Pb[1]);
		
		_flush();
		_unmap_pinned();
	}
	
	_update_clprofiler();
	
	if (bProfile)
	{
		_lets_wait(2, &evEuler3[0], &evEuler3[1]);
		profiler.pop_stop();
		
		profiler.push_start("OpenCL FTLE");
	}
	
	_compute_ftle(FTLEfield, finalT, 2, &evEuler3[0], &evEuler3[1]);
	
	if (bProfileOCL) prof.update();
	if (bProfileOCL && bGFLOPs) _clprofile_summary();
	
	if (bProfile) profiler.pop_stop();

	return FTLEfield;
}
