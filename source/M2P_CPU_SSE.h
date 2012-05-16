/*
 *  M2P_CPU_SSE.h
 *  FTLE2D
 *
 *  Created by Christian Conti on 1/7/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#include "OperatorM2P.h"


class OperatorM2P_CPU_SSE : public OperatorM2P
{
private:
	Timer timer;
	void _m2p_super_aggressive(int sizeX, int sizeY, const float * const x_p, const float * const y_p, const float * const u_mesh, float * const u_p);
	void _m2p_super_aggressive(int sizeX, int sizeY, const double * const x_p, const double * const y_p, const double * const u_mesh, double * const u_p);

public:
	OperatorM2P_CPU_SSE(int sX, int sY);
	void operator()(const Real * const u_m, const Real * const x_p, Real * u_p);
	void operator()(const Real * const u_m, const Real * const x_p, Real * u_p, const int channel);
	double getBandwidth();
	void startTiming();
	void stopTiming();
};
