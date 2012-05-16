/*
 * ProfilerCL.h
 * FTLE2D
 *
 *  Created on: Mar 5, 2010
 *      Author: claurent
 */

#ifndef PROFILERCL_H_
#define PROFILERCL_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <math.h>
#include <map>

#include "CheckCL.h"

class ProfileCLAgent
{
protected:
	cl::Event* event;
	double time;
	cl_ulong accTime;
	int samples;
	
public:
	ProfileCLAgent() {};
	
	ProfileCLAgent(cl::Event* event) : event(event), time(0.0), accTime(0), samples(0) {};
	
	void update()
	{	
		try
		{			
			if (event->getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() != CL_COMPLETE)
			{
				cout << "ProfilerCL error!\n";
				CheckCL(event->getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				abort();
			}
			
			cl_ulong start_time = event->getProfilingInfo<CL_PROFILING_COMMAND_START>();
			cl_ulong stop_time = event->getProfilingInfo<CL_PROFILING_COMMAND_END>();
			
			accTime += stop_time - start_time;
			const double running_time = ((double)stop_time - (double)start_time);
			time += running_time * 1.e-9;
			samples++;
			
			if (start_time>stop_time)
			{
				cout << "WARNING, timing errors!\n";
				cout << "\tStart: " << start_time << endl;
				cout << "\tEnd: " << stop_time << endl;
				cout << "\t" << ((double)stop_time-(double)start_time)*1.e-9 << "s after " << samples << " samples\n";
			}
		}
		catch (cl::Error err) { CheckRelaxCL(err.err()) ; terminate(); }
	}
	
	double getTime()
	{
		return time;
	}
	
	int getSamples()
	{
		return samples;
	}
	
	double getAverage()
	{
		return time/samples;
	}
	
	void printSummary()
	{
		printf("\nAvgTime:\t%e [s]", time / samples);
	}
};

class ProfilerCL
{
private:
	map<string, ProfileCLAgent> m_mapAgents;
	
public:
	
	void addTask(string name, cl::Event* event)
	{
		ProfileCLAgent agent(event);
		m_mapAgents[name] = agent;
	}
	
	void update()
	{
		assert(!m_mapAgents.empty());
		for(map<string, ProfileCLAgent>::iterator it = m_mapAgents.begin(); it != m_mapAgents.end(); it++)
			it->second.update();
	}
	
	double getTime(string name)
	{
		assert(!m_mapAgents.empty());
		
		map<string, ProfileCLAgent>::iterator it = m_mapAgents.find(name.c_str());
		assert(it != m_mapAgents.end());
		
		return it->second.getTime();
	}
	
	double getTotalTime()
	{
		map<string, ProfileCLAgent>::iterator it;
		double totalTime = 0;
		if(m_mapAgents.size() != 0)
			for(it = m_mapAgents.begin(); it != m_mapAgents.end(); it++)
				totalTime += it->second.getTime();
		
		return totalTime;
	}
	
	map<string, ProfileCLAgent> getMapAgents() { return m_mapAgents; }
	
	void print()
	{
		map<string, ProfileCLAgent>::iterator it;
		double totalTime = 0;
		if(m_mapAgents.size() != 0)
			for(it = m_mapAgents.begin(); it != m_mapAgents.end(); it++)
				totalTime += it->second.getTime();
		
		printf("================================================================");
		printf("\nOpenCL events time allocation:\n");
		
		for(it = m_mapAgents.begin(); it != m_mapAgents.end(); it++)
			printf("[%-30s]:\t%-3.3e s\t%3.2f%%\t%d samples\n", it->first.c_str(), it->second.getTime(), it->second.getTime()/totalTime*100, it->second.getSamples());
		printf("[%-30s]:\t%-3.3e s\n", "TOTAL", totalTime);
		
		printf("================================================================\n");
	}
	
	ProfileCLAgent& operator()(string name) { return m_mapAgents[name]; }
};

#endif /* PROFILERCL_H_ */
