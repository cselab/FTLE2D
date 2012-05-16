/*
 *  FTLE_OpenCL_Brunton.h
 *  FTLE2D
 *
 *  Created by Diego Rossinelli on 7/24/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include <map>

using namespace std;

#include "FTLE_OpenCL.h"
#include "BruntonTraversal.h"

class FTLE_OpenCL_Brunton: public FTLE_OpenCL
{
	map<string, ProfilerCL> mapProfilersCL;
	
	cl::CommandQueue cmdq2;
	
	typedef BruntonTraversal::FlowmapID FlowmapID;
	typedef pair<cl::Image2D, cl::Image2D> Flowmap;
	
	ArgumentParser parser;
	bool bTraversalInitialize;
	BruntonTraversal traversal;
	
	map<FlowmapID, Flowmap > fmaps;
	vector<const Real *> ufields, vfields;
	vector<Real> dts;
	
	cl::Kernel kerDisp, kerBruntonX, kerBruntonY;
	cl::Event evDisp[2], evD2F[2], evAdd[2], evInitP[2], evI2B[2], evWriteField[2], evB2I[2];
	
	
	void _dispose(vector<FlowmapID> dead);
	cl::Event _fresh_fine(vector<FlowmapID> fines);
	cl::Event _fresh_coarse(vector<FlowmapID> coarses, cl::Event wait);
	cl::Event _compose(vector<FlowmapID> needed, cl::Buffer result, cl::Event wait);
	
	cl::Event _displacement(cl::Kernel kerDisp, cl::CommandQueue commandqueue, cl::Buffer a, cl::Buffer b, cl::Buffer c, const float dt, const int nevents, ...) const;

	vector<cl::Event> _single_fresh_fine(const int field_id0, const int field_id1, float dt, Flowmap& output, vector<cl::Event> * ev = NULL);
	vector<cl::Event> _single_fresh_coarse(Flowmap& fine0, Flowmap& fine1, Flowmap& output, vector<cl::Event> * ev = NULL);
	
	cl::Event _img2buf(cl::Image2D imgsrc, cl::Buffer bufdest, cl::CommandQueue commandqueue, const int nevents, ...);
	cl::Event _buf2img(cl::Buffer bufsrc, cl::Image2D imgdest, cl::CommandQueue commandqueue, const int nevents, ...);
	cl::Event _m2p_bruntonX(cl::CommandQueue cmdq, cl::Image2D imgField, cl::Buffer bufX, cl::Buffer bufY, cl::Buffer bufResult, const int nevents, ...);
	cl::Event _m2p_bruntonY(cl::CommandQueue cmdq, cl::Image2D imgField, cl::Buffer bufX, cl::Buffer bufY, cl::Buffer bufResult, const int nevents, ...);
	
	Real * _BruntonFTLE(const int nFramesPerFTLE, const Real finalT);

public:
	
	FTLE_OpenCL_Brunton(const int argc, const char ** argv, int sX, int sY, Real dh, bool bProfileOCL, bool bWrapOnHost);
	
	vector<Real *> operator()(vector<Real *>& u, vector<Real>& dt, const Real T);
};