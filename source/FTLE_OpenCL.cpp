/*
 *  FTLE_OpenCL.cpp
 *  libFTLE2D
 *
 *  Created by Diego Rossinelli on 7/12/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#include <cassert>
#include <iomanip>
#include <sstream>

using namespace std;

#include "FTLE_OpenCL.h"
#include "dumpToVTK.h"

void FTLE_OpenCL::_lets_wait(const int nevents, ...)
{
	vector<cl::Event> events;
	
	{
		va_list ev_list;
		
		va_start(ev_list, nevents);
		for(int i=0; i<nevents; i++)
			events.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}
	
	if (events.size() == 0) return;
	
	try { cl::Event::waitForEvents(events); }
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
}

void FTLE_OpenCL::_compute_ftle(float * const destptr, const Real finalT, const int nevents, ...)
{
	vector<cl::Event> events;
	
	{
		va_list ev_list;
		
		va_start(ev_list, nevents);
		for(int i=0; i<nevents; i++)
			events.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}
	
	try
	{
		kerFTLE.setArg(0, bufX);
		kerFTLE.setArg(1, bufY);
		kerFTLE.setArg(2, (float)(1./finalT));
		kerFTLE.setArg(3, bufFTLE);
		
		cmdq.enqueueNDRangeKernel(kerFTLE, cl::NDRange(0), cl::NDRange(gls_ftle), cl::NDRange(wgs_ftle), &events, &evFTLE);
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	try
	{
		vector<cl::Event> waitRead;
		waitRead.push_back(evFTLE);
		
		cmdq.enqueueReadBuffer(bufFTLE, CL_TRUE, 0, size*sizeof(cl_float), FTLEfield, &waitRead, &evReadFTLE);
	} 
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
}

cl::Event FTLE_OpenCL::_load_image(const float * const srcptr, cl::Image2D& imgDest, const bool blocking, cl::CommandQueue commandqueue, vector<cl::Event> events)
{
	try
	{
		if (bWrapOnHost)
		{
			size_t pitch = 0;
			cl::Event retval;
			
			float * const dstptr = (float*)commandqueue.enqueueMapImage(imgDest, CL_TRUE, CL_MAP_WRITE, origin, region, &pitch, 0, &events, &retval);
			
			_FillFieldFromField(srcptr, dstptr);
			
			commandqueue.enqueueUnmapMemObject(imgDest, dstptr);
			
			return retval;
		}
		else
		{
			cl::Event retval;
			
			commandqueue.enqueueWriteImage(imgDest, blocking? CL_TRUE : CL_FALSE, origin, region, 0, 0, (void*)srcptr, &events, &retval);
			
			return retval;
		}
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
}

cl::Event FTLE_OpenCL::_load_image(const float * const srcptr, cl::Image2D& imgDest, const bool blocking, cl::CommandQueue commandqueue, const int numevents, ...)
{
	vector<cl::Event> events;
	
	{
		va_list ev_list;
		
		va_start(ev_list, numevents);
		for(int i=0; i<numevents; i++)
			events.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}

	return _load_image(srcptr, imgDest, blocking, commandqueue, events);
}


cl::Event FTLE_OpenCL::_initpos(cl::Kernel kerInit, cl::CommandQueue commandqueue, cl::Buffer bufResult)
{
	return _initpos(kerInit, commandqueue, bufResult, vector<cl::Event>());
}

cl::Event FTLE_OpenCL::_initpos(cl::Kernel kerInit, cl::CommandQueue commandqueue, cl::Buffer bufResult, vector<cl::Event> events)
{
	cl::Event retval;
	
	kerInit.setArg(0, bufResult);
	commandqueue.enqueueNDRangeKernel(kerInit, cl::NDRange(0), cl::NDRange(gls_euler), cl::NDRange(wgs_euler), &events, &retval);
	
	return retval;
}

cl::Event FTLE_OpenCL::_eu(cl::Kernel kerEu, cl::CommandQueue commandqueue, cl::Buffer a, cl::Buffer b, cl::Buffer c, const float dt, const int nevents, ...) const
{
	vector<cl::Event> events;
	
	{
		va_list ev_list;
		
		va_start(ev_list, nevents);
		for(int i=0; i<nevents; i++)
			events.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}
	
	cl::KernelFunctor k(kerEu, commandqueue, cl::NDRange(0), cl::NDRange(gls_euler), cl::NDRange(wgs_euler));
	
	return k(a, b, c, dt, &events);
}

cl::Event FTLE_OpenCL::_m2p(cl::CommandQueue commandqueue, cl::Image2D f, cl::Buffer x, cl::Buffer y, cl::Buffer r, const int nevents, ...)
{
	vector<cl::Event> events;
	
	{
		va_list ev_list;
		
		va_start(ev_list, nevents);
		for(int i=0; i<nevents; i++)
			events.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}
		
	cl::KernelFunctor k(kerM2P, commandqueue, cl::NDRange(0), cl::NDRange(gls_m2p), cl::NDRange(wgs_m2p));
	
	return  k(f, x, y, r, events.size() == 0 ? NULL : &events); 
}

void FTLE_OpenCL::_initialize()
{
	origin.push_back(0);
	origin.push_back(0);
	origin.push_back(0);
	
	region.push_back(sizeX/4);
	region.push_back(sizeY);
	region.push_back(1);
	
	const int nGPUs = parser("-nGPUs").asInt(1);
	assert(nGPUs == 1);
	
	const int device_rank = parser("-device").asInt(0);
	cmdq = engineCL.getCommandQueue(0, device_rank);
	
	const cl::Device device = cl::Device( cmdq.getInfo<CL_QUEUE_DEVICE>() );

	try
	{
		const string kernel_path = parser("-ftle-kernel-folder").asString("../../FTLE/include/");
		const string path = kernel_path + "FTLE.cl";
		
		stringstream ss;
		ss << "-cl-fast-relaxed-math -D SX="<< sizeX <<" -D SY="<< sizeY << scientific << setprecision(13) << " -D INV2H="<< .5*invh <<"f";
		ss << " -D H=" << h << " -D CELLSHIFT=" << bCentered ? "0.5f" : "0";
		
		if (bVerbose) cout << "building FTLE.cl... options:" << ss.str() << endl;
		
		engineCL.compileProgram(path, ss.str());
		
		kerInitXp = engineCL.getKernel("InitXp", path);
		kerInitYp = engineCL.getKernel("InitYp", path);
		kerEu2 = engineCL.getKernel("Euler2", path);
		kerEu3 = engineCL.getKernel("Euler3", path);
		kerFTLE = engineCL.getKernel("FTLE", path);
		
		wgs_euler = kerEu2.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
		gls_euler = ((sizeX*sizeY + wgs_euler-1)/wgs_euler)*wgs_euler;
		
		wgs_ftle = kerFTLE.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
		gls_ftle = ((sizeX*sizeY + wgs_ftle-1)/wgs_ftle)*wgs_ftle;
	} 
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	try
	{
		const string kernel_path = parser("-ftle-kernel-folder").asString("../../FTLE/include/");
		const string path = kernel_path + "M2P_GPU_single.cl";
		
		stringstream ss;
		ss << "-cl-fast-relaxed-math -D SX="<< sizeX <<" -D SY="<< sizeY <<" -D NOF_ELEMENTS="<< sizeX*sizeY;
		ss << " -D H=" << h << " -D CELLSHIFT=" << bCentered ? "0.5f" : "0";
		
		if (bVerbose) cout << "building M2P_GPU_single.cl... options:" << ss.str() << endl;
		
		engineCL.compileProgram(path, ss.str());
		
		kerM2P = engineCL.getKernel("m2p_mp4", path);
		
		wgs_m2p = kerM2P.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
		gls_m2p = ((sizeX*sizeY + wgs_m2p-1)/wgs_m2p)*wgs_m2p;
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	try
	{
		const cl::ImageFormat format(CL_RGBA,CL_FLOAT);
		const cl::Context ctext = engineCL.getContext();
		const size_t size_bytes = sizeof(float)*sizeX*sizeY;
		
		imgUm =    cl::Image2D(ctext, CL_MEM_READ_ONLY | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR , format, sizeX/4, sizeY);
		imgVm =    cl::Image2D(ctext, CL_MEM_READ_ONLY | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, format, sizeX/4, sizeY);
		bufX =     cl::Buffer(ctext, CL_MEM_READ_WRITE | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, size_bytes);
		bufY =     cl::Buffer(ctext, CL_MEM_READ_WRITE | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, size_bytes);
		bufXstar = cl::Buffer(ctext, CL_MEM_READ_WRITE | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, size_bytes);
		bufYstar = cl::Buffer(ctext, CL_MEM_READ_WRITE | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, size_bytes);
		bufUp =    cl::Buffer(ctext, CL_MEM_READ_WRITE | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, size_bytes);
		bufVp =    cl::Buffer(ctext, CL_MEM_READ_WRITE | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, size_bytes);
		bufFTLE =  cl::Buffer(ctext, CL_MEM_WRITE_ONLY | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, size_bytes);
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	if (bProfileOCL)
	{		
		prof.addTask("FTLE", &evFTLE);
		trajectories_prof.addTask("H2D - unext", &evWriteNextField[0]);
		trajectories_prof.addTask("H2D - vnext", &evWriteNextField[1]);
		trajectories_prof.addTask("M2Pa - u", &evM2Pa[0]);
		trajectories_prof.addTask("M2Pa - v", &evM2Pa[1]);
		trajectories_prof.addTask("M2Pb - u", &evM2Pb[0]);
		trajectories_prof.addTask("M2Pb - v", &evM2Pb[1]);
		trajectories_prof.addTask("Euler2 - u", &evEuler2[0]);
		trajectories_prof.addTask("Euler2 - v", &evEuler2[1]);
		trajectories_prof.addTask("Euler3 - u", &evEuler3[0]);
		trajectories_prof.addTask("Euler3 - v", &evEuler3[1]);
	}
}

void FTLE_OpenCL::_clprofile_summary()
{
	const double GFLOP = (48*1+40)*sizeX*sizeY*1e-9;
	
	const double t[] = { 
		trajectories_prof("M2Pa - u").getAverage(), 
		trajectories_prof("M2Pa - v").getAverage(), 
		trajectories_prof("M2Pb - u").getAverage(), 
		trajectories_prof("M2Pb - v").getAverage() 
	};
	
	const double gflops[] = { GFLOP/t[0], GFLOP/t[1], GFLOP/t[2], GFLOP/t[3]};
	
	const double gflops_avg = .25*(gflops[0] + gflops[1] + gflops[2] + gflops[3]);
	
	printf("Average M2P performance: %.2f GFLOP/s, [%d%% %d%% %d%% %d%%]\n", 
		   gflops_avg, 
		   (int)(100*(gflops[0] - gflops_avg)/gflops_avg),
		   (int)(100*(gflops[1] - gflops_avg)/gflops_avg),
		   (int)(100*(gflops[2] - gflops_avg)/gflops_avg),
		   (int)(100*(gflops[3] - gflops_avg)/gflops_avg));
	
	/*
	 cout << "\nOverlap:\t";
	 
	 double accWallTime = profiler.getAgentTime("OpenCL Advection Loop");
	 double accSerialTime = trajectories_prof.getTotalTime();
	 double overlappingTime = max(0.,accSerialTime-accWallTime);
	 double overhead = max(0.,accWallTime-accSerialTime);
	 
	 
	 cout << "overlap time: " << overlappingTime << "\toverlap percentage: " << max(0.,(accSerialTime/accWallTime-1.)*100.) << "%\n";
	 cout << "visible overhead time: " << overhead << "\tvisible overhead percentage: " << max(0.,(accWallTime/accSerialTime-1.)*100.) << "%\n";
	 cout << "\t(Serial: " << accSerialTime << "s\tWall:" << accWallTime << "s)\n\n";
	 */
}

Real * FTLE_OpenCL::_FTLE(queue<Real *> u, vector<Real> dts, const int frame, const Real finalT)
{
	queue<Real> dt(deque<Real>(dts.begin(), dts.end()));
		
	if (bProfile) profiler.push_start("OpenCL");

	cl::Event initXp = _initpos(kerInitXp, cmdq, bufX);
	cl::Event initYp = _initpos(kerInitYp, cmdq, bufY);	
	
	cl::Event initUm = _load_image(u.front(), imgUm, false, cmdq, 0);
	cl::Event initVm = _load_image(u.front() + sizeX*sizeY, imgVm, false, cmdq, 0);
	
	cmdq.flush();
	
	if (bProfile) profiler.push_start("OpenCL Advection Loop");
	
	const int N = u.size();
	while (u.size() > 1)
	{
		if (bVerbose) printf("iteration %d\n", N-(int)u.size());
		
		const bool bINIT = u.size() == N;
		const float deltat = dt.front();
		
		dt.pop();
		u.pop();
		
		try
		{
			if (bINIT)
			{
				evM2Pa[0] = _m2p(cmdq, imgUm, bufX, bufY, bufUp, 3, &initXp, &initYp, &initUm);
				evM2Pa[1] = _m2p(cmdq, imgVm, bufX, bufY, bufVp, 3, &initXp, &initYp, &initVm);
			}
			else 
			{
				evM2Pa[0] = _m2p(cmdq, imgUm, bufX, bufY, bufUp, 2, &evEuler3[0], &evEuler3[1]);
				evM2Pa[1] = _m2p(cmdq, imgVm, bufX, bufY, bufVp, 2, &evEuler3[0], &evEuler3[1]);
			}
							
			evEuler2[0] = _eu(kerEu2, cmdq, bufX, bufUp, bufXstar, deltat, 1, &evM2Pa[0]);
			evEuler2[1] = _eu(kerEu2, cmdq, bufY, bufVp, bufYstar, deltat, 1, &evM2Pa[1]);
			
			evWriteNextField[0] = _load_image(u.front(), imgUm, false, cmdq, 1, &evM2Pa[0]);
			evWriteNextField[1] = _load_image(u.front() + sizeX*sizeY, imgVm, false, cmdq, 1, &evM2Pa[1]);

			evM2Pb[0] = _m2p(cmdq, imgUm, bufXstar, bufYstar, bufUp, 3, &evEuler2[0], &evEuler2[1], &evWriteNextField[0]);
			evM2Pb[1] = _m2p(cmdq, imgVm, bufXstar, bufYstar, bufVp, 3, &evEuler2[0], &evEuler2[1], &evWriteNextField[1]);

			evEuler3[0] = _eu(kerEu3, cmdq, bufX, bufUp, bufXstar, deltat, 1, &evM2Pb[0]);
			evEuler3[1] = _eu(kerEu3, cmdq, bufY, bufVp, bufYstar, deltat, 1, &evM2Pb[1]);
		}
		catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
		
		cmdq.flush(); 		
		
		if (bProfileOCL)
		{
			_lets_wait(2, &evEuler3[0], &evEuler3[1]);
	
			trajectories_prof.update();
		}
	}
	
	if (bProfile)
	{
		_lets_wait(2, &evEuler3[0], &evEuler3[1]);
		profiler.pop_stop();
	}
	
	_compute_ftle(FTLEfield, finalT, 2, &evEuler3[0], &evEuler3[1]);
	
	if (bProfile) profiler.pop_stop();
	if (bProfileOCL) prof.update();
	
	return FTLEfield;
}


FTLE_OpenCL::FTLE_OpenCL(const int argc, const char ** argv, int sX, int sY, Real dh, EngineCL& engineCL, bool bProfileOCL, bool bWrapOnHost) : 
FTLE2D(argc, argv, sX, sY, dh, false, true), engineCL(engineCL), bProfileOCL(bProfileOCL), bWrapOnHost(bWrapOnHost)
{
	_initialize();
}


FTLE_OpenCL::FTLE_OpenCL(const int argc, const char ** argv, int sX, int sY, Real dh, bool bProfileOCL, bool bWrapOnHost) : 
FTLE2D(argc, argv, sX, sY, dh, false, true), engineCL(EngineCL::getInstance(1, bProfileOCL)), bProfileOCL(bProfileOCL), bWrapOnHost(bWrapOnHost)
{
	_initialize();
}

vector<Real *> FTLE_OpenCL::operator()(vector<Real *>& u, vector<Real>& dt, const Real T)
{
	vector<Real *> result = FTLE2D::operator()(u, dt, T);
	
	if (bProfileOCL) trajectories_prof.print();
	if (bProfileOCL) prof.print();
	if (bProfileOCL) _clprofile_summary();
	
	
	return result;
}