/*
 *  EngineCL.h
 *	FTLE2D
 *
 *  Created by Christian Conti on 06/04/11.
 *
 *	The purpose of this class is to alleviate the programmer
 *		from the burden of setting up the OpenCL framework
 *		and to help him handle the various resources available
 *
 *	Relations between components can be found at page 20 of the OpenCL 1.1 specifications
 *
 *	If allowed by the device, the command queues are set with:
 *		CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
 *		CL_QUEUE_PROFILING_ENABLE
 *
 *	By default, the following choices are made:
 *		activePlatform = platforms[0]
 *		activeDevices = devices[activePlatform][0] (or the 1st nDevices if specified)
 *		activeContext = contexts[0]
 *
 *	The program generation parses for kernel methods
 *		and add them to its list of kernels automatically
 *
 */

#pragma once

#include <assert.h>

#include <vector>
#include <string.h>
#include <map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
using namespace std;

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "cl.hpp"
#include "CheckCL.h"

class EngineCL
{
	const bool bVerbose, bProfile, bOutOfOrder;
	
	static EngineCL * singleton;

	vector<cl::Platform> platforms;
	map<cl_platform_id, vector<cl::Device> > devices;
	vector<cl::Context> contexts;
	map<cl_device_id, vector<cl::CommandQueue> > commandQueues;
	map<cl_context, map<string, cl::Program> > programs;
	//map<cl::Context, map<string, cl::Kernel> > kernels;
	map<string, cl_mem> memObjects;
	
	cl::Platform activePlatform;
	vector<cl::Device> activeDevices;
	cl::Context activeContext;
	
	void _launchError(bool condition, string message);
	void _launchWarning(bool condition, string message);
	
	void _initPlatforms();
	void _initDevices(int nDevices);
	void _initContexts();
	void _initCommandQueues(int nCommandQueuesPerDevice);
		
	string _loadTextFromFile(string filename);
	char* _loadPrecompiledBinaries(string filename, string source, cl::Device device, string options);
	
	// Constructors
	EngineCL(int nDevices=1, int nCommandQueuesPerDevice=1, bool bProfile=false, bool bOutOfOrder=true);
		
public:
	
	EngineCL(const EngineCL& o);
	
	static EngineCL& getInstance(const int numQueuesPerDevice, const bool bProfile=true, const bool bOutOfOrder = true)
	{
		if (singleton == NULL)
			singleton = new EngineCL(-1, numQueuesPerDevice, bProfile, bOutOfOrder);
		
		return * singleton;
	}
	
	// Compile program
	void compileProgram(string programPath, string options="");
	
	//
	void switchPlatform(cl::Platform newPlatform);
	
	// Getters
	vector<cl::Device> getDevices();
	cl::Context getContext(int contextId=0);
	cl::CommandQueue getCommandQueue(int queueId=0, int device=0);
	cl::Kernel getKernel(string kernelName, string programPath);
	
	// Information printers to help debugging of OpenCL code
	void printPlatformInfo();
	void printDeviceInfo(int device);
	void printContextInfo();
	void printCommandQueueInfo(cl_command_queue commandQueue);
	void printProgramInfo(cl_program program);
	void printKernelInfo(string kernelName);
};
