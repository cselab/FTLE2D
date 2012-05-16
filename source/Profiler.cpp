/*
 *  Profiler.cpp
 *  FTLE2D
 *
 *  Created by Diego Rossinelli on 9/13/08.
 *  Copyright 2008 CSE Lab, ETH Zurich. All rights reserved.
 *
 */

#include "Profiler.h"

#ifdef _USE_TBB

#include "tbb/tick_count.h"
using namespace tbb;

void ProfileAgent::_getTime(tick_count& time)
{
	time = tick_count::now();
}

float ProfileAgent::_getElapsedTime(const tick_count& tS, const tick_count& tE)
{
	return (tE - tS).seconds();
}
	
#else
#include <time.h>
void ProfileAgent::_getTime(clock_t& time)
{
	time = clock();
}

float ProfileAgent::_getElapsedTime(const clock_t& tS, const clock_t& tE)
{
	return (tE - tS)/(double)CLOCKS_PER_SEC;
}

#endif
	
