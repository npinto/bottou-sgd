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

#ifndef LOSS_H
#define LOSS_H

#include <cmath>

struct LogLoss
{
  // logloss(a,y) = log(1+exp(-a*y))
  static double loss(double a, double y)
  {
    double z = a * y;
    if (z > 18) 
      return exp(-z);
    if (z < -18)
      return -z;
    return log(1 + exp(-z));
  }
  // -dloss(a,y)/da
  static double dloss(double a, double y)
  {
    double z = a * y;
    if (z > 18) 
      return y * exp(-z);
    if (z < -18)
      return y;
    return y / (1 + exp(z));
  }
};

struct HingeLoss
{
  // hingeloss(a,y) = max(0, 1-a*y)
  static double loss(double a, double y)
  {
    double z = a * y;
    if (z > 1) 
      return 0;
    return 1 - z;
  }
  // -dloss(a,y)/da
  static double dloss(double a, double y)
  {
    double z = a * y;
    if (z > 1) 
      return 0;
    return y;
  }
};

struct SquaredHingeLoss
{
  // squaredhingeloss(a,y) = 1/2 * max(0, 1-a*y)^2
  static double loss(double a, double y)
  {
    double z = a * y;
    if (z > 1)
      return 0;
    double d = 1 - z;
    return 0.5 * d * d;
    
  }
  // -dloss(a,y)/da
  static double dloss(double a, double y)
  {
    double z = a * y;
    if (z > 1) 
      return 0;
    return y * (1 - z);
  }
};

struct SmoothHingeLoss
{
  // smoothhingeloss(a,y) = ...
  static double loss(double a, double y)
  {
    double z = a * y;
    if (z > 1)
      return 0;
    if (z < 0)
      return 0.5 - z;
    double d = 1 - z;
    return 0.5 * d * d;
  }
  // -dloss(a,y)/da
  static double dloss(double a, double y)
  {
    double z = a * y;
    if (z > 1) 
      return 0;
    if (z < 0)
      return y;
    return y * (1 - z);
  }
};

#endif
