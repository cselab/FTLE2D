/*
 *  EngineCL.cpp
 *	FTLE2D
 *
 *  Created by Christian Conti and Diego Rossinelli on 7/11/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#include "EngineCL.h"

EngineCL * EngineCL::singleton = NULL;

#pragma mark Error/Warning handling methods
void EngineCL::_launchError(bool condition, string message)
{
	if (condition)
	{
		cout << "ERROR: " << message << endl;
		abort();
	}
}

void EngineCL::_launchWarning(bool condition, string message)
{
	if (condition)
		cout << "Warning: " << message << endl;
}

#pragma mark Initialization methods for the constructor
void EngineCL::_initPlatforms()
{
	cl::Platform::get(&platforms);
	_launchError(platforms.size()<1, "EngineCL::_initPlatforms(): No OpenCL platform present on the system!");
	
	activePlatform = platforms[0];
	
	if (bVerbose)
	{
		for (int i=0; i<platforms.size(); i++)
		{
			std::string vendor, name, version;
			platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &vendor);
			platforms[i].getInfo((cl_platform_info)CL_PLATFORM_NAME, &name);
			platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &version);
			std::cerr << "Platform " << i << " is by: " << vendor << ", " << name << ", " << version << endl;
		}
	}
}

void EngineCL::_initDevices(int nDevices)
{
	vector<cl::Platform>::iterator platformIter;
	for (platformIter = platforms.begin(); platformIter != platforms.end(); ++platformIter)
	{
		platformIter->getDevices(CL_DEVICE_TYPE_ALL, &devices[(*platformIter)()]);
		
		_launchError(devices.size()<1, "No OpenCL device present on the system!");
		
		// cout devices
		if (bVerbose)
		{
			string namePlatform = platformIter->getInfo<CL_PLATFORM_NAME>();
			cout << "Platform " << namePlatform << endl;
			
			for (int i=0; i<devices.size(); i++)
				string nameDevice = devices[(*platformIter)()][i].getInfo<CL_DEVICE_NAME>();
			
			cout << endl;
		}
	}
	
	activeDevices.clear();
	cout << "Running on: " << endl;
	
	const int M = devices[activePlatform()].size();
	const int N = nDevices>0 ? min(nDevices, M) : M;
	
	for (int i=0; i<N; i++)
	{
		activeDevices.push_back(devices[activePlatform()][i]);
		
		string nameP = activePlatform.getInfo<CL_PLATFORM_NAME>();
		string nameD = devices[activePlatform()][i].getInfo<CL_DEVICE_NAME>();
		
		cout << "\tDevice " << i << ": " << nameP << " - " << nameD << endl;
	}
	
	cout << endl;
	
	_launchWarning(activeDevices.size() < nDevices && nDevices > 0, "Not enough devices found, using all devices found for the current platform");
}

void EngineCL::_initContexts()
{
	cl_int status;
	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)activePlatform(), 0 };
	
	contexts.push_back(cl::Context(activeDevices, properties, NULL, NULL, &status));
	CheckCL(status);
	assert(contexts.size()==1);
	
	activeContext = contexts[0];
}

void EngineCL::_initCommandQueues(int nCommandQueuesPerDevice)
{
	try 
	{
		for (int i=0; i<activeDevices.size(); i++)
		{
			commandQueues[activeDevices[i]()].clear();
			
			cl_command_queue_properties props = activeDevices[i].getInfo<CL_DEVICE_QUEUE_PROPERTIES>() & 
			( CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE * bOutOfOrder | CL_QUEUE_PROFILING_ENABLE * bProfile);
			
			for (int n=0; n<nCommandQueuesPerDevice; n++)
				commandQueues[activeDevices[i]()].push_back(cl::CommandQueue(activeContext, activeDevices[i](), props));
		}
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate();}
}

#pragma mark Program builder helpers
string EngineCL::_loadTextFromFile(string filename)
{
	_launchError(true, "Do not use this!");
	
	fstream filestream(filename.c_str());
	_launchError(filestream==NULL,"Cannot open program file " + filename);
	
	string text;
	while (filestream)
	{
		string buf;
		getline(filestream,buf);
		text += buf;
		text += '\n';
	}
	
	return text;
}

char* EngineCL::_loadPrecompiledBinaries(string filename, string source, cl::Device device, string options)
{
	_launchError(true, "Not implemented yet!");
	
	_launchWarning(true,"currently not caching programs on file!");
	return NULL;
}

#pragma mark Constructors
EngineCL::EngineCL(int nDevices, int nCommandQueuesPerDevice, bool bProfile, bool bOutOfOrder) : 
bVerbose(true), bProfile(bProfile), bOutOfOrder(bOutOfOrder)
{	
	cout << "Starting Engine!\n";
	// setting platforms informations, set first to active
	_initPlatforms();
	
	// setting devices informations
	_initDevices(nDevices);
	
	// create a context for the active devices and the activ e platform
	_initContexts();
	
	// create command queues for the active devices
	_initCommandQueues(nCommandQueuesPerDevice);
	
	// dump system informations
	//_DumpInfoOnFile();
	
	if (bVerbose)
	{
		printPlatformInfo();
		printContextInfo();
		
		cout << "\n===================================================================\n\n";
		
		for (int i=0; i<activeDevices.size(); i++) printDeviceInfo(i);
		
		cout << "\n===================================================================\n\n";
		
		for (int i=0; i<activeDevices.size(); i++)
			for (int q=0; q<commandQueues[activeDevices[i]()].size(); q++)
				printCommandQueueInfo(commandQueues[activeDevices[i]()][q]());
	}
}

EngineCL::EngineCL(const EngineCL& o) :
platforms(o.platforms), devices(o.devices), contexts(o.contexts),
commandQueues(o.commandQueues), programs(o.programs),// kernels(o.kernels),
activePlatform(o.activePlatform), activeDevices(o.activeDevices), activeContext(o.activeContext),
memObjects(o.memObjects), bVerbose(o.bVerbose), bProfile(o.bProfile), bOutOfOrder(o.bOutOfOrder)
{
}

#pragma mark Program
void EngineCL::compileProgram(string programPath, string options)
{
	if (programs[activeContext()].find(programPath) == programs[activeContext()].end())
	{
		
			ifstream file(programPath.c_str());
			
			_launchError(!file.good(), ("Failed to open OpenCL program file: " + programPath).c_str());
			
			string prog(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()) );
			
			cl::Program::Sources source(1, make_pair(prog.c_str(), prog.length()+1));
			cl::Program program(activeContext, source);
		try 
		{
			program.build(activeDevices, options.c_str());
		}
		catch (cl::Error err) 
		{ 
			
				cout << "\n====================================================================\n";
				cout << "\tDumping bugged source code\n\n";
				
				for (int i=0; i<activeDevices.size(); i++)
				{
					cout << "\t\tDevice " << i << endl;
					
					string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(activeDevices[i]);
					
					cout << str << endl << endl;
				}
				
				cout << "\n\tDump completed!";
				cout << "\n====================================================================\n";
				
				_launchError(true, "invalid OpenCL source code");
			
			CheckRelaxCL(err.err()) ; terminate();
		}
		
		programs[activeContext()][programPath] = program;
	}
}

#pragma mark Switch Platform
void EngineCL::switchPlatform(cl::Platform newPlatform)
{
	activePlatform = newPlatform;
	_launchError(true,"This method is incomplete - changing platform means changing context, devices and queues as well");
}

#pragma mark Getters
vector<cl::Device> EngineCL::getDevices()
{
	return activeDevices;
}

cl::Context EngineCL::getContext(int contextId)
{
	_launchError(contextId>=contexts.size(),"This context does not exist!");
	return contexts[contextId];
}

cl::CommandQueue EngineCL::getCommandQueue(int queueId, int device)
{
	_launchError(activeDevices.size()<=device, "This device does not exist or is not active.");
	_launchError(commandQueues[activeDevices[device]()].size()<=queueId, "The requested queue does not exist.");
	
	return commandQueues[activeDevices[device]()][queueId];
}

cl::Kernel EngineCL::getKernel(string kernelName, string programPath)
{
	cl_int status;
	cl::Kernel kernel(programs[activeContext()][programPath], kernelName.c_str(), &status);
	CheckCL(status);
	
	return kernel;
}

#pragma mark Information printers
void EngineCL::printPlatformInfo()
{
	string str;
	
	const int spacing1 = 30;
	const int spacing2 = 100;
	
	cout << "\n===================================================================\n\n";
	cout << "  PLATFORM INFORMATION\n\n";
	
	CheckCL( activePlatform.getInfo(CL_PLATFORM_NAME, &str) );
	cout << left << setw(spacing1) << "\tName:" << left << setw(spacing2) << str.c_str() << endl;
	
	CheckCL( activePlatform.getInfo(CL_PLATFORM_VENDOR, &str) );
	cout << left << setw(spacing1) << "\tVendor:" << left << setw(spacing2) << str.c_str() << endl;
	
	CheckCL( activePlatform.getInfo(CL_PLATFORM_VERSION, &str) );
	cout << left << setw(spacing1) << "\tVersion:" << left << setw(spacing2) << str.c_str() << endl;
	
	CheckCL( activePlatform.getInfo(CL_PLATFORM_EXTENSIONS, &str) );
	cout << left << setw(spacing1) << "\tExtensions:" << left << setw(spacing2) << str.c_str() << endl;
	
	cout << "\n===================================================================\n\n";
}

void EngineCL::printDeviceInfo(int idevice)
{		
	cout << "\n\n\n  DEVICE " << idevice << " INFORMATION\n\n";
	
	const int spacing1 = 30;
	const int spacing2 = 100;
	
	try 
	{
		string name = activeDevices[idevice].getInfo<CL_DEVICE_NAME>();
		string version = activeDevices[idevice].getInfo<CL_DEVICE_VERSION>();
		cl_device_type type = activeDevices[idevice].getInfo<CL_DEVICE_TYPE>();
		cl_uint maxcu = activeDevices[idevice].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		string extensions = activeDevices[idevice].getInfo<CL_DEVICE_EXTENSIONS>();
		size_t memsize =  activeDevices[idevice].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
		size_t totmemsize =  activeDevices[idevice].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
		
		cout << left << setw(spacing1) << "\tDevice Name:" << left << setw(spacing2) << name << endl;
		cout << left << setw(spacing1) << "\tDevice Version:" << left << setw(spacing2) << version << endl;
		cout << left << setw(spacing1) << "\tDevice Type:" << left << setw(spacing2) << (type == CL_DEVICE_TYPE_CPU? "CPU" : (type == CL_DEVICE_TYPE_GPU ? "GPU" : "Accelerator" )) << endl;
		cout << left << setw(spacing1) << "\tMax Compute Units:" << left << setw(spacing2) << maxcu << endl;
		cout << left << setw(spacing1) << "\tDevice Extensions:" << left << setw(spacing2) << extensions << endl;
		cout << left << setw(spacing1) << "\tMax Memory Alloc Size (MB):" << left << setw(spacing2) << memsize/1024./1024. << endl;
		cout << left << setw(spacing1) << "\tGlobal Memory Size (MB):" << left << setw(spacing2) << totmemsize/1024./1024. << endl;
	}
	catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate();}
}

void EngineCL::printContextInfo()
{
}

void EngineCL::printCommandQueueInfo(cl_command_queue commandQueue)
{
}

void EngineCL::printProgramInfo(cl_program program)
{
}

void EngineCL::printKernelInfo(string kernelName)
{
	cl::string str;
	/*
	 cout << "\n===================================================================\n\n";
	 cout << "  KERNEL " << kernelName << " INFORMATION\n\n";
	 CheckCL( kernels[activeContext][kernelName].getInfo(CL_KERNEL_FUNCTION_NAME, &str) );
	 cout << "\tName:\t\t" << str.c_str() << endl;
	 CheckCL( kernels[activeContext][kernelName].getInfo(CL_KERNEL_NUM_ARGS, &n) );
	 cout << "\tNumber of Arguments:\t" << n << endl;
	 cout << "\n===================================================================\n\n";
	 */
}
