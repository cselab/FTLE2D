/*
 *  Profiler.h
 *  FTLE2D
 *
 *  Created by Diego Rossinelli on 9/13/08.
 *  Copyright 2008 CSE Lab, ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <assert.h>
#undef min
#undef max
#include <vector>
#undef min
#undef max
#include <map>
#include <string>
#include <stack>
#include <cstdio>

using std::map;
using std::vector;
using std::string;
using std::stack;

#include <sys/time.h>
class MTTimer
{
	struct timeval t_start, t_end;
	struct timezone t_zone;
	
public:
	
	void start()
	{
		gettimeofday(&t_start,  &t_zone);
	}
	
	double stop()
	{
		gettimeofday(&t_end,  &t_zone);
		return (t_end.tv_usec  - t_start.tv_usec)*1e-6  + (t_end.tv_sec  - t_start.tv_sec);
	}
};


#ifdef _USE_TBB
#include "tbb/tick_count.h"
namespace tbb { class tick_count; }
#else
#include <time.h>
#endif

#pragma once

const bool bVerboseProfiling = false;
	
class ProfileAgent
{
#ifdef _USE_TBB
	typedef tbb::tick_count ClockTime;
#else
	typedef clock_t ClockTime;
#endif
	
	enum ProfileAgentState{ ProfileAgentState_Created, ProfileAgentState_Started, ProfileAgentState_Stopped};
	
	MTTimer timer;
	ClockTime m_tStart, m_tEnd;
	ProfileAgentState m_state;
	double m_dAccumulatedTime;
	int m_nMeasurements;
	double m_nMoney;
	
	static void _getTime(ClockTime& time);
	static float _getElapsedTime(const ClockTime& tS, const ClockTime& tE);
	
	void _reset()
	{
		m_tStart = ClockTime();
		m_tEnd = ClockTime();
		m_dAccumulatedTime = 0;
		m_nMeasurements = 0;
		m_nMoney = 0;
		m_state = ProfileAgentState_Created;
	}
	
public:
	
	ProfileAgent():m_tStart(), m_tEnd(), m_state(ProfileAgentState_Created),
	m_dAccumulatedTime(0), m_nMeasurements(0), m_nMoney(0) {}
	
	
	void start()
	{
		assert(m_state == ProfileAgentState_Created || m_state == ProfileAgentState_Stopped);
		
		if (bVerboseProfiling) {printf("start\n");}
		
		timer.start();
		
		m_state = ProfileAgentState_Started;
	}
	
	void stop(double nMoney=0)
	{
		assert(m_state == ProfileAgentState_Started);
		
		if (bVerboseProfiling) {printf("stop\n");}
		
		m_dAccumulatedTime += timer.stop();
		m_nMeasurements++;
		m_nMoney += nMoney;
		m_state = ProfileAgentState_Stopped;
	}
	
	friend class Profiler;
};
	
struct ProfileSummaryItem
{
	string sName;
	double dTime;
	double dAverageTime;
	double nMoney;
	int nSamples;
	
	ProfileSummaryItem(string sName_, double dTime_, double nMoney_, int nSamples_): 
		sName(sName_), dTime(dTime_), nMoney(nMoney_),nSamples(nSamples_), dAverageTime(dTime_/nSamples_){}
};


class Profiler
{
protected:

protected:
	
	map<string, ProfileAgent*> m_mapAgents;
	stack<string> m_mapStoppedAgents;
	
public:
	void push_start(string sAgentName)
	{
		if (m_mapStoppedAgents.size() > 0)
			getAgent(m_mapStoppedAgents.top()).stop();
		
		m_mapStoppedAgents.push(sAgentName);
		getAgent(sAgentName).start();
	}
	
	void pop_stop(double dMoney=0)
	{
		string sCurrentAgentName = m_mapStoppedAgents.top();
		getAgent(sCurrentAgentName).stop(dMoney);
		m_mapStoppedAgents.pop();
		
		if (bVerboseProfiling) printf("%s is running now...\n",m_mapStoppedAgents.top().data());
		
		if (m_mapStoppedAgents.size() == 0) return;
		
		getAgent(m_mapStoppedAgents.top()).start();
	}
	
	
	void clear()
	{
		for(map<string, ProfileAgent*>::iterator it = m_mapAgents.begin(); it != m_mapAgents.end(); it++)
		{
			delete it->second;
			
			it->second = NULL;
		}
		
		m_mapAgents.clear();
	}
	
	Profiler(): m_mapAgents(){}
	
	~Profiler()
	{
		clear();
	}
	
	void printSummary() const
	{
		assert(m_mapStoppedAgents.size()==0);
		vector<ProfileSummaryItem> v = createSummary();
		
		double dTotalTime = 0;
		for(vector<ProfileSummaryItem>::const_iterator it = v.begin(); it!= v.end(); it++)
			dTotalTime += it->dTime;
		
		for(vector<ProfileSummaryItem>::const_iterator it = v.begin(); it!= v.end(); it++)
		{
			const ProfileSummaryItem& item = *it;
			printf("[%-25s]: \t%02.2f%%\t%03.6f s\t%8d samples\n",
				   item.sName.data(), 100*item.dTime/dTotalTime, item.dTime, item.nSamples);
		}
		
		printf("[Total time]: \t%f\n", dTotalTime);
	}
	
	vector<ProfileSummaryItem> createSummary() const
	{
		vector<ProfileSummaryItem> result;
		result.reserve(m_mapAgents.size());
		
		for(map<string, ProfileAgent*>::const_iterator it = m_mapAgents.begin(); it != m_mapAgents.end(); it++)
		{
			const ProfileAgent& agent = *it->second;
			
			result.push_back(ProfileSummaryItem(it->first, agent.m_dAccumulatedTime, agent.m_nMoney, agent.m_nMeasurements));
		}
		
		return result;
	}
	
	void reset()
	{
		for(map<string, ProfileAgent*>::const_iterator it = m_mapAgents.begin(); it != m_mapAgents.end(); it++)
			it->second->_reset();
	}
	

	
	ProfileAgent& getAgent(string sName)
	{
		if (bVerboseProfiling) {printf("%s ", sName.data());}
		
		map<string, ProfileAgent*>::const_iterator it = m_mapAgents.find(sName);
		
		const bool bFound = it != m_mapAgents.end();
		
		if (bFound) return *it->second;

		ProfileAgent * agent = new ProfileAgent();
		
		m_mapAgents[sName] = agent;
		
		return *agent;		
	}
	
	double getAgentTime(string sName)
	{
		ProfileAgent agent = getAgent(sName);
		
		return agent.m_dAccumulatedTime;
	}
	
	friend class ProfileAgent;
};



