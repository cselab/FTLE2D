/*
 *  ProblemSettings.h
 *  FTLE2D
 *
 *  Created by Christian Conti on 5/16/12.
 *  Copyright 2012 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <iomanip>

using namespace std;

void runProblem(FTLE2D * ftle2d, const int argc, const char ** argv)
{	
	ArgumentParser parser(argc, argv);
	
	parser.set_strict_mode();
	
	vector<Real *> nvel;
	vector<Real> dt;
	
	const int sx = parser("-sx").asInt();
	const int sy = parser("-sy").asInt();
	const int nFields = parser("-nFields").asInt();
	const int nFTLEs = parser("-nFTLEs").asInt();
	double h = 1./sx;
	//double T_FTLE = 0.;
	double t = 0.;
	double eps = .25;
	double A = .1;
	double omega = M_PI*.2;
	double delta = .25*h;
	Real * velocity;
	//const Real u = 1.;
	//const Real a = .2;
	const Real ratio = sy/sx;
	
	double maxV = 0.;
	
	for (int i=0; i<nFields; i++)
	{
		velocity = (Real*)_mm_malloc(sizeof(Real)*sx*sy*2, 16);
		const Real at = eps*sin(omega*t);
		const Real bt = 1.-2.*eps*sin(omega*t);

#pragma omp parallel for
		for (int iy=0; iy<sy; iy++)
			for (int ix=0; ix<sx; ix++)
			{
				// Time-independent Double Gyre
				//Real x = ix/(Real)sy;
				//Real y = iy/(Real)sy;
				//velocity[ix+sx*iy]       = -sin(M_PI*x) * cos(M_PI*y) * M_PI;
				//velocity[ix+sx*iy+sx*sy] = +sin(M_PI*y) * cos(M_PI*x) * M_PI;
				
				/*
				 Real x = 2.*ix/(Real)sx-1.;
				 x = x==0?h*.5:x;
				 Real y = 2.*iy/(Real)sy-1.;
				 y = y==0?h*.5:y;
				 const Real x2 = x*x;
				 const Real y2 = y*y;
				 const Real r2 = x2+y2;
				 Real sinT = y/(x*sqrt(y2/x2+1));
				 Real cosT = 1./(sqrt(y2/x2+1));
				 const Real fP = 1.+a*a/r2;
				 const Real fM = 1.-a*a/r2;
				 velocity[ix+sx*iy] = u*fM*cosT*cosT+u*fP*sinT*sinT;
				 velocity[ix+sx*iy+sx*sy] = u*fM*cosT*sinT-u*fP*cosT*sinT;
				 //*/
				//*
				 const Real f = at*ix*ix*h*h*4 + bt*ix*h*2;
				 const Real dfdx = 2*at*ix*h*2 + bt;
				 velocity[ix+sx*iy	   ] = -M_PI*A*sin(M_PI*f)*cos(M_PI*iy*h*2);
				 velocity[ix+sx*iy+sx*sy] =  M_PI*A*cos(M_PI*f)*sin(M_PI*iy*h*2)*dfdx;
				 //velocity[ix+sx*iy	   ] = -M_PI*A*sin(omega/3.*2.*t)*sin(M_PI*f)*cos(M_PI*iy*h*2);
				 //velocity[ix+sx*iy+sx*sy] =  M_PI*A*sin(t*f)*cos(M_PI*f)*sin(M_PI*iy*h*2)*dfdx;
				 //*/
			}
		nvel.push_back(velocity);
		dt.push_back(delta);
		t += delta;
	}
	
	dump("velocity.vti", 0, nvel[0],nvel[0]+sx*sy, sx, sy, true);

	Timer timer;
	
	FTLE2D& ftle = *ftle2d;
	
	const int spacing1 = 30;
	const int spacing2 = 100;
	
	cout << "FTLE Computation\n";
	cout << left << setw(spacing1) << "\tFTLE length\t" << left << setw(spacing2) << (nFields-nFTLEs+1) << endl;
	cout << left << setw(spacing1) << "\tnFTLEs\t\t\t" << left << setw(spacing2) << nFTLEs << endl;
	cout << left << setw(spacing1) << "\tnFields\t\t" << left << setw(spacing2) << nFields << endl;
	timer.start();
	vector<Real *> res1 = ftle(nvel, dt, (nFields-nFTLEs)*delta);
	printf("Time:	%f\n",timer.stop());
	
	parser.unset_strict_mode();
	printf("Data Dumping\n");
	timer.start();
#pragma omp parallel for
	for (int i=0; i<res1.size(); i++)
		dump("ftleTest_%03d.vti", i, res1[i], sx, sy, false);
	printf("Time:	%f\n",timer.stop());
}
