/*
 *  FTLE.cl
 *  FTLE2D
 *
 *  Created by Christian Conti on 1/26/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */
 
#define GID get_global_id(0)

kernel void InitXp(	global write_only float * xp )
{
	xp[GID] = (CELLSHIFT + GID % SX) * H;
}

kernel void InitYp( global write_only float * yp)
{
	yp[GID] = (CELLSHIFT + GID / SX) * H;
}

kernel void Displacement(	global float * xp,
							global float const * up,
							global float const * xp_star,
							float dt)
{
	xp[GID] = (xp_star[GID] + xp[GID] + dt * up[GID])*0.5f - xp[GID];
}

kernel void Euler2(	global float const * xp,
					global float const * up,
					global write_only float * xp_star,
					float dt)
{
	xp_star[GID] = xp[GID] + dt * up[GID];
}

kernel void Euler3(	global float * xp,
					global float const * up,
					global float const * xp_star,
					float dt)
{
	xp[GID] = (xp_star[GID] + xp[GID] + dt * up[GID])*0.5f;
}
 
kernel void FTLE(	global float const * particlesPosX,
					global float const * particlesPosY,
					float invT,
					global write_only float * ftle)
{
	const int ix = GID%SX;
	const int iy = GID/SX;
	
	if (ix<1 || ix>=SX-1 || iy<1 || iy>=SY-1) return;
	
	const float j00 = INV2H*(particlesPosX[GID+ 1] - particlesPosX[GID- 1]);
	const float j01 = INV2H*(particlesPosX[GID+SX] - particlesPosX[GID-SX]);
	const float j10 = INV2H*(particlesPosY[GID+ 1] - particlesPosY[GID- 1]);
	const float j11 = INV2H*(particlesPosY[GID+SX] - particlesPosY[GID-SX]);

	const float t1 = j10;
	const float t2 = t1 * t1;
	const float t3 = j11;
	const float t4 = t3 * t3;
	const float t5 = j00;
	const float t6 = t5 * t5;
	const float t7 = j01;
	const float t8 = t7 * t7;
	const float t9 = t2 * t2;
	const float t16 = t4 * t4;
	const float t21 = t6 * t6;
	const float t24 = t8 * t8;
	const float t29 = t9 + 2.f * (t4 * t2 + t6 * t2 - t2 * t8 + t16 - t4 * t6 + t8 * t4 + t21 + t6 * t8 + t24 + 4.f * t5 * t1 * t7 * t3);
	const float t30 = sqrt(t29);
	const float lambda = (t2 + t4 + t6 + t8 + t30) * .5f;
	
	ftle[get_global_id(0)] = .5f*invT*log(lambda);
}
