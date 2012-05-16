/*
 *  OperatorM2P.h
 *  FTLE2D
 *
 *  Created by Christian Conti on 1/7/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <sys/time.h>
#include <iostream>

#ifndef _DOUBLE
typedef float Real;
#else
typedef double Real;
#endif



class Timer
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

class OperatorM2P
{
protected:
	int sizeX, sizeY;
	double time;
	
public:
	OperatorM2P(int sX, int sY) : sizeX(sX), sizeY(sY), time(0.) {}
	
	virtual void operator()(const Real * const u_m, const Real * const x_p, Real * u_p) = 0;
	virtual void operator()(const Real * const u_m, const Real * const x_p, Real * u_p, const int channel) { std::cout << "not there yet\n"; abort(); }
	double getTime() { return time; }
	double getPerformance() { return (48*2+40)*sizeX*sizeY/time/1.e9; }
	virtual double getBandwidth() = 0;
	virtual void startTiming() = 0;
	virtual void stopTiming() = 0;
};