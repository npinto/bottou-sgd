// -*- C++ -*-
// A simple timer.
// Copyright (C) 2007- Leon Bottou

// This library is free software; you can redistribute it and/or
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


#include "timer.h"
#include <ctime>

#ifdef USE_REALTIME_CLOCK
# include <sys/time.h>
# include <time.h>
static double
klock()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double) tv.tv_sec + (double) tv.tv_usec * 1e-6;
  return (double) std::clock() / (double) CLOCKS_PER_SEC;
}
#else
static double
klock()
{
  return (double) std::clock() / (double) CLOCKS_PER_SEC;
}
#endif

Timer::Timer()
  : a(0), s(0), r(0)
{
}

void 
Timer::reset()
{
  a = 0;
  s = 0;
  r = 0;
}


double 
Timer::elapsed()
{
  double n = klock();
  if (r)
    a += n - s;
  s = n;
  return a;
}

double 
Timer::start()
{
  elapsed();
  r = 1;
  return a;
}



double
Timer::stop()
{
  elapsed();
  r = 0;
  return a;
}




