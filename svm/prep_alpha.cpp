// -*- C++ -*-
// SVM with stochastic gradient (preprocessing)
// Copyright (C) 2007- Leon Bottou

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA


#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

#include "assert.h"
#include "vectors.h"
#include "gzstream.h"

using namespace std;

#define DATADIR "../data/pascal/"
#define DATAFILE "alpha"
#define DATATEST  250000

typedef vector<SVector> xvec_t;
typedef vector<double> yvec_t;

int 
load(const char *fname, xvec_t &xp, yvec_t &yp)
{
  cerr << "# Reading " << fname << endl;
  ifstream f(fname);
  if (! f.good())
    assertfail("Cannot open file " << fname);
  int count = 0;
  while (f.good())
    {
      double y;
      SVector x;
      f >> y >> x;
      if (f.good())
        {
          xp.push_back(x);
          yp.push_back(y);
          count += 1;
        }
    }
  if (! f.eof())
    assertfail("Failed reading " << fname);
  cerr << "# Done reading " << count << " examples." << endl;
  return count;
}

void
saveBinary(const char *fname, xvec_t &xp, yvec_t &yp, 
           vector<int> &index, int imin, int imax)
{
  cerr << "# Writing " << fname << endl;
  ogzstream f;
  f.open(fname);
  if (! f.good())
    assertfail("ERROR: cannot open " << fname << " for writing.");
  int count = 0;
  for (int ii=imin; ii<imax; ii++)
    {
      int i = index[ii];
      double y = yp[i];
      SVector x = xp[i];
      f.put((y >= 0) ? 1 : 0);
      x.save(f);
      count += 1;
    }
  cerr << "# Wrote " << count << " examples." << endl;
}


int main(int, const char**)
{
  // load data
  vector<SVector> xp;
  vector<double> yp;
  int count = load(DATADIR DATAFILE ".txt", xp, yp);
  // compute random shuffle
  cerr << "# Shuffling" << endl;
  vector<int> index(count);
  for (int i=0; i<count; i++) index[i] = i;
  random_shuffle(index.begin(), index.end());
  random_shuffle(index.begin(), index.end());
  // saving
  saveBinary(DATAFILE ".test.bin.gz", xp, yp, index, 0, DATATEST);  
  saveBinary(DATAFILE ".train.bin.gz", xp, yp, index, DATATEST, count);
}
