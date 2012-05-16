/*
 *  FTLE_OpenCL_Brunton.cpp
 *  FTLE2D
 *
 *  Created by Diego Rossinelli on 7/24/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#include <vector>
#include <map>
#include <algorithm>

using namespace std;

#include "FTLE_OpenCL_Brunton.h"
#include "dumpToVTK.h"

cl::Event FTLE_OpenCL_Brunton::_m2p_bruntonX(cl::CommandQueue cmdq, cl::Image2D imgField, cl::Buffer bufX, cl::Buffer bufY, cl::Buffer bufResult, const int nevents, ...)
{
	vector<cl::Event> events;
	
	{
		va_list ev_list;
		
		va_start(ev_list, nevents);
		for(int i=0; i<nevents; i++)
			events.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}
	
	cl::KernelFunctor k(kerBruntonX, cmdq, cl::NDRange(0), cl::NDRange(gls_m2p), cl::NDRange(wgs_m2p));
	
	return  k(imgField, bufX, bufY, bufResult, events.size() == 0 ? NULL : &events); 
}

cl::Event FTLE_OpenCL_Brunton::_m2p_bruntonY(cl::CommandQueue cmdq, cl::Image2D imgField, cl::Buffer bufX, cl::Buffer bufY, cl::Buffer bufResult, const int nevents, ...)
{
	vector<cl::Event> events;
	
	{
		va_list ev_list;
		
		va_start(ev_list, nevents);
		for(int i=0; i<nevents; i++)
			events.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}
	
	cl::KernelFunctor k(kerBruntonY, cmdq, cl::NDRange(0), cl::NDRange(gls_m2p), cl::NDRange(wgs_m2p));
	
	return  k(imgField, bufX, bufY, bufResult, events.size() == 0 ? NULL : &events); 
}

cl::Event FTLE_OpenCL_Brunton::_displacement(cl::Kernel kerDisp, cl::CommandQueue commandqueue, cl::Buffer a, cl::Buffer b, cl::Buffer c, const float dt, const int nevents, ...) const
{
	vector<cl::Event> events;
	
	{
		va_list ev_list;
		
		va_start(ev_list, nevents);
		for(int i=0; i<nevents; i++)
			events.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}
	
	cl::KernelFunctor k(kerDisp, commandqueue, cl::NDRange(0), cl::NDRange(gls_euler), cl::NDRange(wgs_euler));
	
	return k(a, b, c, dt, &events);
}

void FTLE_OpenCL_Brunton::_dispose(vector<FlowmapID> dead)
{
	if (bVerbose) cout << "Map size before disposing is: " << fmaps.size() << endl;
	for(vector<FlowmapID>::const_iterator it=dead.begin(); it!=dead.end(); ++it)
	{
		if (bVerbose) cout << "Removing (" << it->first << "," << it->second << ")\n";
		fmaps.erase(*it);
	}
}

cl::Event FTLE_OpenCL_Brunton::_img2buf(cl::Image2D imgsrc, cl::Buffer bufdest, cl::CommandQueue commandqueue, const int nevents, ...)
{
	vector<cl::Event> vwait;
	
	{
		va_list ev_list;
		
		va_start(ev_list, nevents);
		for(int i=0; i<nevents; i++)
			vwait.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}
	
	cl::Event retval;
	try
	{
		commandqueue.enqueueCopyImageToBuffer(imgsrc, bufdest, origin, region, 0, &vwait, &retval);
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	return retval;
}

cl::Event FTLE_OpenCL_Brunton::_buf2img(cl::Buffer bufsrc, cl::Image2D imgdest, cl::CommandQueue commandqueue, const int nevents, ...)
{
	vector<cl::Event> vwait;
	
	{
		va_list ev_list;
		
		va_start(ev_list, nevents);
		for(int i=0; i<nevents; i++)
			vwait.push_back(*va_arg(ev_list, cl::Event*));
		va_end(ev_list);
	}
	
	cl::Event retval;
	try
	{
		commandqueue.enqueueCopyBufferToImage(bufsrc, imgdest, 0, origin, region, &vwait, &retval);
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	return retval;
}

vector<cl::Event> FTLE_OpenCL_Brunton::_single_fresh_fine(const int field_id0, const int field_id1, float deltat, Flowmap& output, vector<cl::Event> * ev)
{
	vector<cl::Event> wait;
	if (ev != NULL) wait = *ev;
	
	vector<cl::Event> retval;
	try
	{
		evInitP[0] = _initpos(kerInitXp, cmdq,  bufX, wait);
		cmdq.flush();
		evInitP[1] = _initpos(kerInitYp, cmdq2, bufY, wait);
		
		evWriteField[0] = _load_image(ufields[field_id0], imgUm, false, cmdq,  1, &evInitP[0]);
		cmdq.flush();
		cmdq2.flush();
		evWriteField[1] = _load_image(vfields[field_id0], imgVm, false, cmdq2, 1, &evInitP[1]);
		
		evI2B[0] = _img2buf(imgUm, bufUp, cmdq,  1, &evWriteField[0]);
		cmdq.flush();
		cmdq2.flush();
		evI2B[1] = _img2buf(imgVm, bufVp, cmdq2, 1, &evWriteField[1]);
		
		evEuler2[0] = _eu(kerEu2, cmdq,  bufX, bufUp, bufXstar, deltat, 2, &evInitP[0], &evI2B[0]);
		cmdq.flush();
		cmdq2.flush();
		evEuler2[1] = _eu(kerEu2, cmdq2, bufY, bufVp, bufYstar, deltat, 2, &evInitP[1], &evI2B[1]);
		
		evWriteNextField[0] = _load_image(ufields[field_id1], imgUm, false, cmdq,  1, &evI2B[0]);
		cmdq.flush();
		cmdq2.flush();
		evWriteNextField[1] = _load_image(vfields[field_id1], imgVm, false, cmdq2, 1, &evI2B[1]);
		
		evM2Pb[0] = _m2p(cmdq,  imgUm, bufXstar, bufYstar, bufUp, 3, &evEuler2[0], &evEuler2[1], &evWriteNextField[0]);
		cmdq.flush();
		cmdq2.flush();
		evM2Pb[1] = _m2p(cmdq2, imgVm, bufXstar, bufYstar, bufVp, 3, &evEuler2[0], &evEuler2[1], &evWriteNextField[1]);
		
		evEuler3[0] = _displacement(kerDisp, cmdq,  bufX, bufUp, bufXstar, deltat, 1, &evM2Pb[0]);
		cmdq.flush();
		cmdq2.flush();
		evEuler3[1] = _displacement(kerDisp, cmdq2, bufY, bufVp, bufYstar, deltat, 1, &evM2Pb[1]);
		
		evB2I[0] = _buf2img(bufX, output.first,  cmdq,  1, &evEuler3[0]);
		cmdq.flush();
		cmdq2.flush();
		evB2I[1] = _buf2img(bufY, output.second, cmdq2, 1, &evEuler3[1]);
		
		retval.push_back(evB2I[0]);
		retval.push_back(evB2I[1]);
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	return retval;
}

cl::Event FTLE_OpenCL_Brunton::_fresh_fine(vector<FlowmapID> fines)
{
	//create new opencl flowmaps
	try
	{
		const cl::ImageFormat format(CL_RGBA,CL_FLOAT);
		const cl::Context ctext = engineCL.getContext();
		
		for(vector<FlowmapID>::const_iterator it=fines.begin(); it!=fines.end(); ++it)
		{
			if (!(fmaps.find(*it)!=fmaps.end()))
			{
				cl::Image2D xphi =  cl::Image2D(ctext, CL_MEM_READ_ONLY | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, format, sizeX/4, sizeY);
				cl::Image2D yphi =  cl::Image2D(ctext, CL_MEM_READ_ONLY | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, format, sizeX/4, sizeY);
				
				fmaps[*it] = Flowmap(xphi, yphi);
			}
		}
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	//compute new flowmaps
	vector<cl::Event> evNew, evOld;
	for(int i=0; i<fines.size(); i++)
	{
		assert(fines[i].first == 0);
		if (bVerbose) cout << "Adding (" << fines[i].first << "," << fines[i].second << ")\n";
		
		const int field_id0 = fines[i].second;
		const int field_id1 = fines[i].second+1;
		
		assert(field_id1 < ufields.size());
		
		evNew = _single_fresh_fine(field_id0, field_id1, dts[field_id0], fmaps[fines[i]], i>0 ? &evOld : NULL);

		evOld = evNew;
		
		if (bProfileOCL)
		{
			cmdq.flush();
			cmdq.finish();
			cmdq2.flush();
			cmdq2.finish();
			mapProfilersCL["ComputeFlowMaps"].update();
		}
	}
	
	cl::Event retval;
	cmdq.enqueueMarker(&retval);
	
	return retval;
}

cl::Event FTLE_OpenCL_Brunton::_fresh_coarse(vector<FlowmapID> coarses, cl::Event wait)
{
	//create fresh opencl flowmaps
	try
	{
		const cl::ImageFormat format(CL_RGBA,CL_FLOAT);
		const cl::Context ctext = engineCL.getContext();
		
		for(vector<FlowmapID>::const_iterator it=coarses.begin(); it!=coarses.end(); ++it)
		{
			if (!(fmaps.find(*it)!=fmaps.end()))
			{
				cl::Image2D xphi =  cl::Image2D(ctext, CL_MEM_READ_ONLY | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, format, sizeX/4, sizeY);
				cl::Image2D yphi =  cl::Image2D(ctext, CL_MEM_READ_ONLY | bWrapOnHost*CL_MEM_ALLOC_HOST_PTR, format, sizeX/4, sizeY);
				
				fmaps[*it] = Flowmap(xphi, yphi);
			}
		}
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	vector<int> levels;
	map<int, vector<int> > level2fmap;
	
	//fill levels and leve2fmap
	{
		for(vector<FlowmapID>::const_iterator it=coarses.begin(); it!=coarses.end(); ++it)
		{
			assert(it->first > 0);
			
			level2fmap[it->first].push_back(it->second);
		}
		
		for(map<int, vector<int> >::const_iterator it = level2fmap.begin(); it!=level2fmap.end(); ++it)
		{
			assert(level2fmap.find(it->first -1) != level2fmap.end() || it->first == 1);
			levels.push_back(it->first);
		}
		
		sort(levels.begin(), levels.end());
	}
	
	
	//compute coarse opencl flowmaps
	vector<cl::Event> evNew, evOld;
	evOld.push_back(wait);
	
	int idx = 0;
	for(vector<int>::const_iterator itl = levels.begin(); itl!=levels.end(); ++itl)
	{
		vector<int> myids = level2fmap[*itl];
		for(vector<int>::const_iterator it = myids.begin(); it!= myids.end(); ++it)
		{
			const int ldest = *itl;
			const int xdest = *it;
			
			if (bVerbose) cout << "Merging (" << ldest-1 << "," << xdest*2 << ")+(" << ldest-1 << "," << xdest*2+1 << ") to (" << ldest << "," << xdest << ")\n";
			
			evNew = _single_fresh_coarse(fmaps[FlowmapID(ldest-1, xdest*2)], fmaps[FlowmapID(ldest-1, xdest*2 + 1)], fmaps[FlowmapID(ldest, xdest)], &evOld);
			
			evOld = evNew;
			
			if (bProfileOCL)
			{
				cmdq.flush();
				cmdq.finish();
				cmdq2.flush();
				cmdq2.finish();
				mapProfilersCL["BuildTree"].update();
			}
		}
	}
	
	cl::Event retval;
	cmdq.enqueueMarker(&retval);
	
	return retval;
	
}

vector<cl::Event> FTLE_OpenCL_Brunton::_single_fresh_coarse(Flowmap& fine0, Flowmap& fine1, Flowmap& output, vector<cl::Event> * ev)
{
	assert(ev != NULL);
	
	vector<cl::Event> retval;
	try
	{
		if (ev->size() == 1)
		{
			cl::Event w0 = (*ev)[0];
			
			evI2B[0] = _img2buf(fine0.first,  bufX, cmdq, 1, &w0);
			evI2B[1] = _img2buf(fine0.second, bufY, cmdq, 1, &w0);
		}
		else 
		{
			assert(ev->size() == 2);
			
			cl::Event w0 = (*ev)[0];
			cl::Event w1 = (*ev)[1];
			
			evI2B[0] = _img2buf(fine0.first,  bufX, cmdq, 1, &w0);
			evI2B[1] = _img2buf(fine0.second, bufY, cmdq, 1, &w1);
		}
		
		evM2Pa[0] = _m2p_bruntonX(cmdq, fine1.first,  bufX, bufY, bufUp, 2, &evI2B[0], &evI2B[1]);
		evM2Pa[1] = _m2p_bruntonY(cmdq, fine1.second, bufX, bufY, bufVp, 2, &evI2B[0], &evI2B[1]);
		
		evB2I[0] = _buf2img(bufUp, output.first,  cmdq, 1, &evM2Pa[0]);
		evB2I[1] = _buf2img(bufVp, output.second, cmdq, 1, &evM2Pa[1]);
		
		retval.push_back(evB2I[0]);
		retval.push_back(evB2I[1]);
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	return retval;	
}

cl::Event FTLE_OpenCL_Brunton::_compose(vector<FlowmapID> needed, cl::Buffer result, cl::Event wait)
{
	cl::Buffer X = bufX, Xnext = bufUp, Y = bufY, Ynext = bufVp;
	
	if (bVerbose) cout << "Collapsing: ";
	try
	{
		evI2B[0] = _img2buf(fmaps[needed.front()].first,  X, cmdq, 1, &wait);
		evI2B[1] = _img2buf(fmaps[needed.front()].second, Y, cmdq, 1, &wait);
		cl::Event ev0 = evI2B[0];
		cl::Event ev1 = evI2B[1];
		
		if (bVerbose) cout << "(" << needed.front().first << "," << needed.front().second << ")";
		
		for(int i=1; i<needed.size(); ++i)
		{
			if (bVerbose) cout << " + (" << needed[i].first << "," << needed[i].second << ")";
			
			evM2Pa[0] = _m2p_bruntonX(cmdq, fmaps[needed[i]].first,  X, Y, Xnext, 2, &ev0, &ev1);
			evM2Pa[1] = _m2p_bruntonY(cmdq, fmaps[needed[i]].second, X, Y, Ynext, 2, &ev0, &ev1);
			
			ev0 = evM2Pa[0];
			ev1 = evM2Pa[1];
			
			swap(X, Xnext);
			swap(Y, Ynext);
			
			
			if (bProfileOCL)
			{
				cmdq.flush();
				cmdq.finish();
				cmdq2.flush();
				cmdq2.finish();
				mapProfilersCL["CollapseFlowMaps"].update();
			}
		}
		
		if (bVerbose) cout << endl << endl;
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	
	bufX = X;
	bufY = Y;
	
	cl::Event retval;
	cmdq.enqueueMarker(&retval);
	
	vector<cl::Event> waitlist;
	waitlist.push_back(retval);
	
	return retval;
	
}

Real * FTLE_OpenCL_Brunton::_BruntonFTLE(const int nFramesPerFTLE, const Real finalT)
{	
	if (!bTraversalInitialize)
	{
		parser.set_strict_mode();
		
		const int nFields = parser("-nFields").asInt();
		const int nFTLEs = parser("-nFTLEs").asInt();
		
		if (bVerbose) cout << "Brunton Traversal with nFields=" << nFields-1 << " nFTLEs=" << nFTLEs << " fields/FTLE=" << nFramesPerFTLE << endl;
		traversal = BruntonTraversal(nFields-1, nFTLEs, nFramesPerFTLE);
		bTraversalInitialize = true;
	}
	
	//TASKS
	//0. get traversal info
	//1. get rid of the dead fmaps
	//2. create the fresh fine fmaps
	//3. create the fresh coarse fmaps
	//4. compose fmaps
	//5. ftle
	
	//0.
	BruntonTraversal::Step stepinfo = traversal.next();
	
	//1.
	if (bProfile) profiler.push_start("0. Dispose");
	_dispose(stepinfo.dead);
	
	//2.
	if (bProfile)
	{
		cmdq.flush();
		cmdq.finish();
		profiler.pop_stop();
		profiler.push_start("1. Compute Flow Maps");
	}
	cl::Event evFF = _fresh_fine(stepinfo.freshfine);
	
	//3.
	if (bProfile)
	{
		cmdq.flush();
		cmdq.finish();
		profiler.pop_stop();
		profiler.push_start("2. Build Tree");
	}
	cl::Event evFC = _fresh_coarse(stepinfo.freshcoarse, evFF);
	
	//4.
	if (bProfile)
	{
		cmdq.flush();
		cmdq.finish();
		profiler.pop_stop();
		profiler.push_start("3. Collapse Flow Maps");
	}
	cl::Event evFinalFlowmap = _compose(stepinfo.needed, bufFTLE, evFC);
	
	//5.
	if (bProfile)
	{
		cmdq.flush();
		cmdq.finish();
		profiler.pop_stop();
		profiler.push_start("4. Compute FTLE");
	}
	_compute_ftle(FTLEfield, finalT, 1, &evFinalFlowmap);
	if (bProfile) profiler.pop_stop();
	if (bProfileOCL)
	{
		cmdq.flush();
		cmdq.finish();
		cmdq2.flush();
		cmdq2.finish();
		mapProfilersCL["FTLE"].update();
	}
	
	return FTLEfield;
}

FTLE_OpenCL_Brunton::FTLE_OpenCL_Brunton(const int argc, const char ** argv, int sX, int sY, Real dh, bool bProfileOCL, bool bWrapOnHost):
FTLE_OpenCL(argc, argv, sX, sY, dh, EngineCL::getInstance(2, bProfileOCL), bProfileOCL, false), bTraversalInitialize(false), parser(argc, argv)

{
	try
	{
		const int device_rank = parser("-device").asInt(0);
		cmdq2 = engineCL.getCommandQueue(1, device_rank);
		
		{
			const string kernel_path = parser("-ftle-kernel-folder").asString("../../FTLE/include/");
			const string path = kernel_path + "FTLE.cl";
			
			kerDisp = engineCL.getKernel("Displacement", path);
		}
		
		{
			const string kernel_path = parser("-ftle-kernel-folder").asString("../../FTLE/include/");
			const string path = kernel_path + "M2P_GPU_single.cl";
			
			kerBruntonX = engineCL.getKernel("xbrunton", path);
			kerBruntonY = engineCL.getKernel("ybrunton", path);
		}
		
		if (bProfileOCL)
		{
			mapProfilersCL["ComputeFlowMaps"].addTask("Init - x", &evInitP[0]);
			mapProfilersCL["ComputeFlowMaps"].addTask("Init - y", &evInitP[1]);
			mapProfilersCL["ComputeFlowMaps"].addTask("H2D - u", &evWriteField[0]);
			mapProfilersCL["ComputeFlowMaps"].addTask("H2D - v", &evWriteField[1]);
			mapProfilersCL["ComputeFlowMaps"].addTask("I2B - u", &evI2B[0]);
			mapProfilersCL["ComputeFlowMaps"].addTask("I2B - v", &evI2B[1]);
			mapProfilersCL["ComputeFlowMaps"].addTask("Euler2 - x", &evEuler2[0]);
			mapProfilersCL["ComputeFlowMaps"].addTask("Euler2 - y", &evEuler2[1]);
			mapProfilersCL["ComputeFlowMaps"].addTask("H2D - u2", &evWriteNextField[0]);
			mapProfilersCL["ComputeFlowMaps"].addTask("H2D - v2", &evWriteNextField[1]);
			mapProfilersCL["ComputeFlowMaps"].addTask("M2P - u", &evM2Pb[0]);
			mapProfilersCL["ComputeFlowMaps"].addTask("M2P - v", &evM2Pb[1]);
			mapProfilersCL["ComputeFlowMaps"].addTask("Euler (Displ) - x", &evEuler3[0]);
			mapProfilersCL["ComputeFlowMaps"].addTask("Euler (Displ) - y", &evEuler3[1]);
			mapProfilersCL["ComputeFlowMaps"].addTask("B2I - u", &evB2I[0]);
			mapProfilersCL["ComputeFlowMaps"].addTask("B2I - v", &evB2I[1]);
			
			mapProfilersCL["BuildTree"].addTask("B2I - u", &evI2B[0]);
			mapProfilersCL["BuildTree"].addTask("B2I - v", &evI2B[1]);
			mapProfilersCL["BuildTree"].addTask("M2P (Displ) - u", &evM2Pa[0]);
			mapProfilersCL["BuildTree"].addTask("M2P (Displ) - v", &evM2Pa[1]);
			mapProfilersCL["BuildTree"].addTask("B2I - u", &evB2I[0]);
			mapProfilersCL["BuildTree"].addTask("B2I - v", &evB2I[1]);
			
			mapProfilersCL["CollapseFlowMaps"].addTask("B2I - u", &evB2I[0]);
			mapProfilersCL["CollapseFlowMaps"].addTask("B2I - v", &evB2I[1]);
			mapProfilersCL["CollapseFlowMaps"].addTask("M2P (Displ) - u", &evM2Pa[0]);
			mapProfilersCL["CollapseFlowMaps"].addTask("M2P (Displ) - v", &evM2Pa[1]);
			
			mapProfilersCL["FTLE"].addTask("FTLE", &evFTLE);
			mapProfilersCL["FTLE"].addTask("D2H - FTLE", &evReadFTLE);
		}
	} 
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
}

vector<Real *> FTLE_OpenCL_Brunton::operator()(vector<Real *>& u, vector<Real>& dt, const Real T)
{
	// assume: time intervals between velocity fields are not "irregular"
	assert(u.size() == dt.size());
	dts = dt;
	assert(T>0.);
	
	if (u.size() != dt.size() + 1)
	{
		if (bVerbose) cout << "Warning! The condition #dts = u.size() - 1 is not satisfied!" << endl << "#dts = " << dts.size() << ", u.size() = " << u.size() << endl;
		if (bVerbose) cout << "We impose #dts := u.size() - 1" << endl << endl;
	}
	
	const int N = u.size();
	for(int i=0; i<N; i++)	
	{
		Real * p = u[i];
		ufields.push_back(p);
		vfields.push_back(p + sizeX*sizeY);
	}
	assert( ufields.size() == N );
	
	vector<Real *> result;
	
	Real sum_dt = 0;
	
	int iFrameCounter = 0;
	int e = 0;
	
	// prepare 1st batch of velocity fields
	do
	{
		if (e >= u.size()-1) break;
		
		sum_dt += dt[e];
		
		e++;
	}
	while (sum_dt+1e-6 < T);
	
	
	bool bContinue = true;
	while (bContinue)
	{
		if (bVerbose) printf("FTLE Frame %d, nFields/FTLE = %d\n", iFrameCounter, e-iFrameCounter);
		
		result.push_back(_BruntonFTLE(e-iFrameCounter, T));
		
		sum_dt -= dt[iFrameCounter];
		
		do
		{
			bContinue = !(e>=u.size()-1);
			
			if (!bContinue) break;
			
			sum_dt += dt[e];
			
			e++;
		}
		while (sum_dt+1e-6 < T);
		
		iFrameCounter++;
	}
	
	if (bProfile) profiler.printSummary();
	if (bProfile) profiler.clear();
	
	if (bProfileOCL)
	{
		mapProfilersCL["ComputeFlowMaps"].print();
		mapProfilersCL["BuildTree"].print();
		mapProfilersCL["CollapseFlowMaps"].print();
		mapProfilersCL["FTLE"].print();
	}
	
	return result;
}