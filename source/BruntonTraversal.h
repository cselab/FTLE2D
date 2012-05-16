/*
 *  BruntonTraversal.h
 *  FTLE2D
 *
 *  Created by Diego Rossinelli on 7/22/11.
 *  Copyright 2011 ETH Zurich. All rights reserved.
 *
 */

#include <cassert>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

class BruntonTraversal
{
public:
	typedef pair<int, int> FlowmapID; // a flowmap is identified by a level and an index
	
	struct Step
	{
		vector< FlowmapID > needed; //flowmaps that are composed together
		vector< FlowmapID > dead; //old flowmaps that wont be used anymore
		vector< FlowmapID > freshfine; //new fine integrations all entries must have level=0
		vector< FlowmapID > freshcoarse; //flow map to be collapsed
	};
	
private:
	const int N; // number of time samples
	const int M; // number of ftles
	const int NLEVELS; //levels in the time-h tree
	const int NperFTLE; //number of fields necessary to generate one ftle
	
	int currFTLE; //id denoting the first velfields used in the current ftle
	vector<int> wstart, wend; //start and end of the windows
	
	//estimate the max number of levels in the time-h flowmaps tree
	int _levels(int N)
	{
		int l = 0;
		
		while(N != 0) 
		{			
			l++;
			N /= 2;
		}
		
		return l;
	}
	
	vector< FlowmapID > _from_to(int s, int e, int l)
	{
		assert(e>=s);
		vector< pair<int, int> > result(e-s);
		
		for(int i=s; i<e; i++)
			result[i-s] = pair<int, int>(l, i);
		
		return result;
	}
	
	//find the cheapest composition paths of flowmaps
	vector< FlowmapID > _find(int s, int e, int l)
	{
		assert(e >= s);
		
		if ( e==s ) return vector< FlowmapID >();
		if ( l==0 ) return _from_to(s, e, l);
		
		const int h = 1 << l;
		const int ks = (s + h-1) / h;
		const int ke = e / h;
		
		if (ks >= ke)
			return _find(s, e, l-1);
		else
		{
			vector< FlowmapID > left = _find(s, ks*h, l-1);
			vector< FlowmapID > center = _from_to(ks, ke, l);
			vector< FlowmapID > right = _find(ke*h, e, l-1);
			
			center.insert(center.begin(), left.begin(), left.end());
			center.insert(center.end(), right.begin(), right.end());
			
			return center;
		}
	}
	
public:
	
	BruntonTraversal(const int nvelfields, const int nftles, const int fields_per_ftle): 
	N(nvelfields), M(nftles), NLEVELS(_levels(nvelfields)), currFTLE(0), NperFTLE(fields_per_ftle)
	{
		wstart = vector<int> (NLEVELS);
		wend = vector<int> (NLEVELS);
	}
	
	BruntonTraversal(): N(-1), M(-1), NLEVELS(-1), NperFTLE(-1){}
	
	const BruntonTraversal& operator=(const BruntonTraversal& c)
	{
		if (&c != this)
		{
			*const_cast<int*>(&N) = c.N;
			*const_cast<int*>(&M) = c.M;
			*const_cast<int*>(&NLEVELS) = c.NLEVELS;
			*const_cast<int*>(&NperFTLE) = c.NperFTLE;
			
			currFTLE = c.currFTLE; 
			wstart = c.wstart;
			wend = c.wend;
		}
		
		return *this;
	}
	
	Step next()
	{
		Step result;
		
		result.needed = _find(currFTLE, currFTLE + NperFTLE, NLEVELS-1);
		
		//fill to_erase
		{
			map<int, int> imin;
			
			for(vector<FlowmapID>::iterator it = result.needed.begin(); it!=result.needed.end(); ++it)
			{
				const bool found = (imin.find(it->first) != imin.end());
				imin[it->first] = min(it->second, found? imin[it->first] : it->second);
			}
			
			for(int l=NLEVELS-1; l>0; l--)
			{
				if (imin.find(l) == imin.end()) continue;
				
				const bool found = (imin.find(l-1) != imin.end());
				
				const int v = 2*imin[l];
				
				imin[l-1] = min(found ? imin[l-1] : v, v);
			}
			
			for(int l=0; l<NLEVELS; l++)
			{
				if (imin.find(l) != imin.end()) continue;
				
				imin[l] = imin[l-1]/2 + 1; //why? well, that was the last chance to jump on it
			}
			
			for(map<int, int>::iterator it = imin.begin(); it!=imin.end(); ++it)
			{
				const int l = it->first;
				
				const int newstart = max(it->second, wstart[l]);
				
				for (int i=wstart[l]; i<newstart; i++)
					result.dead.push_back( FlowmapID(l, i) );
				
				wstart[l] = newstart;
			}
		}
		
		//fill vel2fmap and fmap2fmap
		{		
			map<int, int> imax;
			
			for(vector<FlowmapID>::const_iterator it=result.needed.begin(); it!=result.needed.end(); ++it)
			{
				const bool found = (imax.find(it->first) != imax.end());
				imax[it->first] = max(it->second, found? imax[it->first] : it->second);
			}
			
			for(int l=NLEVELS-1; l>0; l--)
			{
				if (imax.find(l) == imax.end()) continue;
				
				imax[l-1] = max(imax[l-1], 2*imax[l]+1); // why? well, we depend on it
			}
			
			
			for(map<int, int>::iterator it = imax.begin(); it!=imax.end(); ++it)
			{
				const int l = it->first;
				
				const int newend = max(it->second+1, wend[l]);
				
				if (newend == wend[l]) continue;
				
				vector< FlowmapID > v = _from_to(wend[l], newend, l);
				
				if (l == 0)
					result.freshfine.insert(result.freshfine.end(), v.begin(), v.end());
				else
					result.freshcoarse.insert(result.freshcoarse.end(), v.begin(), v.end());
				
				
				wend[l] = newend;
			}
		}
		
		currFTLE++;
		
		return result;
	}
};
