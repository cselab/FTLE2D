/*
 *  main.cpp
 *  FTLE2D
 *
 *  Created by Christian Conti on 1/20/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#include <iostream>
#include <vector>
#include <xmmintrin.h>
#include "FTLE.h"
#include "FTLE2D.h"

#ifdef _OPENCL
#include "FTLE_OpenCL.h"
#include "FTLE_OpenCL_OverlapGPU.h"
#include "FTLE_OpenCL_Brunton.h"
#endif


#include "dumpToVTK.h"
#include "ProblemSettings.h"

using namespace std;


#ifdef _LIBRARY
vector<Real *> libCore(const int argc, const char ** argv, int sx, int sy, Real dh, vector<Real *>& u, vector<Real>& dt, const Real T)
{
	ArgumentParser parser(argc, argv);
	
	if (parser("-help").asBool())
	{
		cout << "usage:\n";
		cout << "\t-help\t\t\t\tdisplay this message\n";
		cout << "\t-verbose [0|1]\t\t\tset to verbose (default: false)\n";
		
		cout << "\t-sx N\t\t\t\tset problem size in x direction\n";
		cout << "\t-sy N\t\t\t\tset problem size in y direction\n";
		cout << "\t-CellCentered [0|1]\t\tdata is cell centered (default: false)\n";
		cout << "\t-nFields N\t\t\tset number of velocity fields\n";
		cout << "\t-nFTLEs N\t\t\tset number of desired FTLE fields-1\n";
		//cout << "\t-FromFile filenamePattern\t\t\tread velocity fields from file\n";
		
		// opencl stuff
		cout << "\t-OpenCL [0|1]\t\t\trun OpenCL implementation (default: false)\n";
		cout << "\t-device N\t\t\trun on device rank N (default: 0)\n";
		//cout << "\t-nGPUs N\t\t\trun on N OpenCL devices (default: 1)\n";
		cout << "\t--ftle-kernel-folder path\tset path to OpenCL kernel files (default: ../../FTLE/include/)\n";
		cout << "\t-type [brunton|overlap]\t\tselect method (default: fallback)\n";
		cout << "\t-onhost [0|1]\t\t\tyse CL_MEM_ALLOC_HOST_PTR for image allocation, only for brunton (default: false)\n";
		
		// profiling stuff
		cout << "\t-profilecl [0|1]\t\tprint OpenCL profiling information (default: false)\n";
		cout << "\t-GFLOPs [0|1]\t\t\tprint GFLOPs information (default: false)\n";
		exit(0);
	}

	FTLE2D * ftle = new FTLE2D(argc, argv, sx, sy, 1./sx, parser("-verbose").asBool());
	
#ifdef _OPENCL
	if (parser("-OpenCL").asBool())
	{
		delete ftle;
		
		if (parser("-type").asString() == "brunton")
		{
			ftle = new FTLE_OpenCL_Brunton(argc, argv, sx, sy, 1./sx, parser("-profilecl").asBool(), parser("-onhost").asBool());
			printf("=========================>>> type is: Brunton\n");
		}
		else if (parser("-type").asString() == "overlap")
		{
			ftle = new FTLE_OpenCL_OverlapGPU(argc, argv, sx, sy, 1./sx, parser("-profilecl").asBool());
			printf("=========================>>> type is: overlap\n");
		}
		else 
		{
			ftle = new FTLE_OpenCL(argc, argv, sx, sy, 1./sx, false, false);
			printf("=========================>>> type is: fallback\n");
		}
	}
#endif
	
	vector<Real *> res = (*ftle)(u, dt, T);

	return res;
}
#else
int main (int argc, const char ** argv)
{
	ArgumentParser parser(argc, argv);
	
	if (parser("-help").asBool())
	{
		cout << "usage:\n";
		cout << "\t-help\t\t\t\tdisplay this message\n";
		cout << "\t-verbose [0|1]\t\t\tset to verbose (default: false)\n";
		
		cout << "\t-sx N\t\t\t\tset problem size in x direction\n";
		cout << "\t-sy N\t\t\t\tset problem size in y direction\n";
		cout << "\t-CellCentered [0|1]\t\tdata is cell centered (default: false)\n";
		cout << "\t-nFields N\t\t\tset number of velocity fields\n";
		cout << "\t-nFTLEs N\t\t\tset number of desired FTLE fields-1\n";
		//cout << "\t-FromFile filenamePattern\t\t\tread velocity fields from file\n";
		
		// opencl stuff
		cout << "\t-OpenCL [0|1]\t\t\trun OpenCL implementation (default: false)\n";
		cout << "\t-device N\t\t\trun on device rank N (default: 0)\n";
		//cout << "\t-nGPUs N\t\t\trun on N OpenCL devices (default: 1)\n";
		cout << "\t--ftle-kernel-folder path\tset path to OpenCL kernel files (default: ../../FTLE/include/)\n";
		cout << "\t-type [brunton|overlap]\t\tselect method (default: fallback)\n";
		cout << "\t-onhost [0|1]\t\t\tyse CL_MEM_ALLOC_HOST_PTR for image allocation, only for brunton (default: false)\n";
		
		// profiling stuff
		cout << "\t-profilecl [0|1]\t\tprint OpenCL profiling information (default: false)\n";
		cout << "\t-GFLOPs [0|1]\t\t\tprint GFLOPs information (default: false)\n";
		exit(0);
	}
	
	const int sx = parser("-sx").asInt();
	const int sy = parser("-sy").asInt();

	
	FTLE2D * ftle = new FTLE2D(argc, argv, sx, sy, 1./sx, parser("-verbose").asBool());
	
#ifdef _OPENCL
	if (parser("-OpenCL").asBool(false))
	{
		delete ftle;
		
		if (parser("-type").asString() == "brunton")
		{
			ftle = new FTLE_OpenCL_Brunton(argc, argv, sx, sy, 1./sx, parser("-profilecl").asBool(), parser("-onhost").asBool());
			printf("=========================>>> type is: Brunton\n");
		}
		else if (parser("-type").asString() == "overlap")
		{
			ftle = new FTLE_OpenCL_OverlapGPU(argc, argv, sx, sy, 1./sx, parser("-profilecl").asBool());
			printf("=========================>>> type is: overlap\n");
		}
		else 
		{
			ftle = new FTLE_OpenCL(argc, argv, sx, sy, 1./sx, parser("-profilecl").asBool(), parser("-onhost").asBool());
			printf("=========================>>> type is: fallback\n");
		}
	}
#endif
	
	runProblem(ftle, argc, argv);
	
    return 0;
}
#endif
