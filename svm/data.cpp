// -*- C++ -*-
// SVM with stochastic gradient
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


#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gzstream.h"
#include "assert.h"
#include "data.h"

using namespace std;

static void
load_datafile_sub(istream &f, bool binary, const char *fname, 
                  xvec_t &xp, yvec_t &yp, int &maxdim,
                  bool normalize, int maxrows)
{
  cout << "# Reading file " << fname << endl;
  if (! f.good())
    assertfail("Cannot open " << fname);
  int ncount = 0;
  int pcount = 0;
  while (f.good() && maxrows--)
    {
      SVector x;
      double y;
      if (binary)
        {
          y = (f.get()) ? +1 : -1;
          x.load(f);
        }
      else
        {
          f >> y >> x;
        }
      if (f.good())
        {
          if (normalize)
            {
              double d = dot(x,x);
              if (d > 0 && d != 1.0)
                x.scale(1.0 / sqrt(d)); 
            }
          if (y != +1 && y != -1)
            assertfail("Label should be +1 or -1.");
          xp.push_back(x);
          yp.push_back(y);
          if (y > 0)
            pcount += 1;
          else
            ncount += 1;
          if (x.size() > maxdim)
            maxdim = x.size();
        }
    }
  cout << "# Read " << pcount << "+" << ncount 
       << "=" << pcount + ncount << " examples." << endl;
}


void
load_datafile(const char *fname, 
              xvec_t &xp, yvec_t &yp, int &maxdim,
              bool normalize, int maxrows)
{
  bool binary = false;
  bool compressed = false;
  string filename = fname;
  int len = filename.size();
  if (len > 7 && filename.substr(len-7) == ".txt.gz")
    compressed = true;
  else if (len > 7 && filename.substr(len-7) == ".bin.gz")
    compressed = binary = true;
  else if (len > 4 && filename.substr(len-4) == ".bin")
    binary = true;
  else if (len > 4 && filename.substr(len-4) == ".txt")
    binary = false;
  else
    assertfail("Filename suffix should be one of: "
               << ".bin, .txt, .bin.gz, .txt.gz");
  if (compressed)
    {
      igzstream f;
      f.open(fname);
      return load_datafile_sub(f, binary, fname, xp, yp, 
                               maxdim, normalize, maxrows);
    }
  else
    {
      ifstream f;
      f.open(fname);
      return load_datafile_sub(f, binary, fname, xp, yp, 
                               maxdim, normalize, maxrows);
    }
}


