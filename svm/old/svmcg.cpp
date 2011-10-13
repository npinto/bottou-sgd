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



// $Id$


#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>

using namespace std;

typedef vector<SVector> xvec_t;
typedef vector<double> yvec_t;


// Select loss
#ifndef LOSS
# define LOSS LOGLOSS
#endif

// Magic to find loss name
#define _NAME(x) #x
#define _NAME2(x) _NAME(x)
const char *lossname = _NAME2(LOSS);

// Available losses
#define HINGELOSS 1
#define SMOOTHHINGELOSS 2
#define SQUAREDHINGELOSS 3
#define LOGLOSS 10
#define LOGLOSSMARGIN 11

// Add bias at index zero during load.
#define REGULARIZEDBIAS 1


inline 
double loss(double z)
{
#if LOSS == LOGLOSS
  if (z > 18)
    return exp(-z);
  if (z < -18)
    return -z;
  return log(1+exp(-z));
#elif LOSS == LOGLOSSMARGIN
  if (z > 18)
    return exp(1-z);
  if (z < -18)
    return 1-z;
  return log(1+exp(1-z));
#elif LOSS == SMOOTHHINGELOSS
  if (z < 0)
    return 0.5 - z;
  if (z < 1)
    return 0.5 * (1-z) * (1-z);
  return 0;
#elif LOSS == SQUAREDHINGELOSS
  if (z < 1)
    return 0.5 * (1 - z) * (1 - z);
  return 0;
#elif LOSS == HINGELOSS
  if (z < 1)
    return 1 - z;
  return 0;
#else
# error "Undefined loss"
#endif
}


inline 
double dloss(double z)
{
#if LOSS == LOGLOSS
  if (z > 18)
    return exp(-z);
  if (z < -18)
    return 1;
  return 1 / (exp(z) + 1);
#elif LOSS == LOGLOSSMARGIN
  if (z > 18)
    return exp(1-z);
  if (z < -18)
    return 1;
  return 1 / (exp(z-1) + 1);
#elif LOSS == SMOOTHHINGELOSS
  if (z < 0)
    return 1;
  if (z < 1)
    return 1-z;
  return 0;
#elif LOSS == SQUAREDHINGELOSS
  if (z < 1)
    return (1 - z);
  return 0;
#else
  if (z < 1)
    return 1;
  return 0;
#endif
}



// -- conjugate gradient

class SvmCg
{
public:
  SvmCg(int dim, double lambda, int trainsize);
  void train(int imin, int imax, const xvec_t &x, const yvec_t &y,
             const char *prefix = "");
  void test(int imin, int imax, const xvec_t &x, const yvec_t &y, 
            const char *prefix = "");
private:
  double  lambda;
  FVector w;
  FVector g;
  FVector u;

  int n;
  FVector ywx;
  FVector yux;
  double ww;
  double wu;
  double uu;

  double search(double tol=1e-4);
  double f(double t);

  double dsearch(double tol=1e-4);
  double df(double t);
};



SvmCg::SvmCg(int dim, double l, int trainsize)
  : lambda(l), w(dim), n(trainsize)
{
  ywx.resize(n);
  yux.resize(n);
}


double 
SvmCg::f(double t)
{
  double cost = 0;
  for (int i=0; i<n; i++)
    cost += loss( ywx[i] + t * yux[i] );
  double norm = ww + 2 * t * wu + t * t * uu;
  return 0.5 * lambda * norm + cost / n;
}


double 
SvmCg::df(double t)
{
  double dcost = 0;
  for (int i=0; i<n; i++)
    dcost += dloss( ywx[i] + t * yux[i] ) * yux[i];
  double dnorm = wu + t * uu;
  return - lambda * dnorm + dcost / n;
}


double 
SvmCg::dsearch(double tol)
{
  double a = 0;
  double fa = df(a);
  double b = 1;
  double fb = df(b);
  if (fa < 0)
    return -1;
  while (fb > 0)
    {
      double ofb = fb;
      b = b * 2;
      assert(b < 1e80);
      fb = df(b);
      if (fb > ofb)
        break;
    }
  if (fb > 0)
    return -1;
  tol *= b - a;
  double e = b - a;
  double d = e;
  while (b - a > 2 * tol && fa - fb > 0)
    {
      double m = (a + b) / 2;
      double x = (fa * b - fb * a) / (fa - fb);
      if (x > a && x < b && fabs(x - m) < fabs(e))
        { e = d / 2; d = x - m; }
      else
        { x = m; }

      double fx = df(x);
      if (fx > 0)
        { fa = fx; a = x; }
      else if (fx < 0)
        { fb = fx; b = x; }
      else
        return x;
    }
  return (a + b) / 2;
}


void 
SvmCg::train(int imin, int imax, 
              const xvec_t &xp, const yvec_t &yp,
              const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  assert(n == imax - imin + 1);
  
  FVector oldg = g;
  g.clear();
  g.add(w, -lambda);
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double z = ywx[i-imin];
#if LOSS < LOGLOSS
      if (z < 1)
#endif
        {
          cost += loss(z);
          g.add(x, dloss(z) * y / n);
        }
    }
  ww= dot(w,w);
  cost = 0.5 * lambda * ww + cost / n;

  if (u.size())
    {
      // conjugate gradient
      oldg.add(g, -1);
      double beta = - dot(g, oldg) / dot(u, oldg);
      u.combine(beta, g, 1);
    }
  else
    {
      // first iteration
      u = g;
    }
  // line search and step
  wu = dot(w,u);
  uu = dot(u,u);
  cout << prefix << setprecision(6) 
       << "Before: ww=" << ww 
       << ", uu=" << uu
       << ", cost=" << cost << endl;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      yux[i-imin] = y * dot(u,x);
    }
  double eta = dsearch();
  if (eta < 0)
    {
      cout << "*** Restarting CG" << endl;
      u.clear();
    }
  else
    {
      w.add(u, eta);
      ywx.add(yux, eta);
    }
}


void 
SvmCg::test(int imin, int imax, 
             const xvec_t &xp, const yvec_t &yp, 
             const char *prefix)

{
  cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  int nerr = 0;
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double wx = dot(w,x);
      double z = y * wx;
      if (z <= 0)
        nerr += 1;
#if LOSS < LOGLOSS
      if (z < 1)
#endif
        cost += loss(z);
    }
  int n = imax - imin + 1;
  double wnorm =  dot(w,w);
  double loss = cost / n;
  cost = loss + 0.5 * lambda * wnorm;
  cout << prefix << setprecision(4)
       << "Misclassification: " << (double)nerr * 100.0 / n << "%." << endl;
  cout << prefix << setprecision(12) 
       << "Cost: " << cost << "." << endl;
  cout << prefix << setprecision(12) 
       << "Loss: " << loss << "." << endl;
}




// --- options

string trainfile;
string testfile;
double lambda = 1e-4;
int epochs = 100;
int trainsize = -1;

void 
usage()
{
  cerr << "Usage: svmsgd [options] trainfile [testfile]" << endl
       << "Options:" << endl
       << " -lambda <lambda>" << endl
       << " -epochs <epochs>" << endl
       << " -trainsize <n>" << endl
       << endl;
  exit(10);
}

void 
parse(int argc, const char **argv)
{
  for (int i=1; i<argc; i++)
    {
      const char *arg = argv[i];
      if (arg[0] != '-')
        {
          if (trainfile.empty())
            trainfile = arg;
          else if (testfile.empty())
            testfile = arg;
          else
            usage();
        }
      else
        {
          while (arg[0] == '-') arg += 1;
          string opt = arg;
          if (opt == "lambda" && i+1<argc)
            {
              lambda = atof(argv[++i]);
              cout << "Using lambda=" << lambda << "." << endl;
              assert(lambda>0 && lambda<1e4);
            }
          else if (opt == "epochs" && i+1<argc)
            {
              epochs = atoi(argv[++i]);
              cout << "Going for " << epochs << " epochs." << endl;
              assert(epochs>0 && epochs<1e6);
            }
          else if (opt == "trainsize" && i+1<argc)
            {
              trainsize = atoi(argv[++i]);
              assert(trainsize > 0);
            }
          else
            usage();
        }
    }
  if (trainfile.empty())
    usage();
}


// --- loading data

int dim;
xvec_t xtrain;
yvec_t ytrain;
xvec_t xtest;
yvec_t ytest;

void
load(const char *fname, xvec_t &xp, yvec_t &yp)
{
  cout << "Loading " << fname << "." << endl;
  
  igzstream f;
  f.open(fname);
  if (! f.good())
    {
      cerr << "ERROR: cannot open " << fname << "." << endl;
      exit(10);
    }
  int pcount = 0;
  int ncount = 0;

  bool binary;
  string suffix = fname;
  if (suffix.size() >= 7)
    suffix = suffix.substr(suffix.size() - 7);
  if (suffix == ".dat.gz")
    binary = false;
  else if (suffix == ".bin.gz")
    binary = true;
  else
    {
      cerr << "ERROR: filename should end with .bin.gz or .dat.gz" << endl;
      exit(10);
    }

  while (f.good())
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
#if REGULARIZEDBIAS
      x.set(0,1);
#endif
      if (f.good())
        {
          assert(y == +1 || y == -1);
          xp.push_back(x);
          yp.push_back(y);
          if (y > 0)
            pcount += 1;
          else
            ncount += 1;
          if (x.size() > dim)
            dim = x.size();
        }
      if (trainsize > 0 && xp.size() > (unsigned int)trainsize)
        break;
    }
  cout << "Read " << pcount << "+" << ncount 
       << "=" << pcount + ncount << " examples." << endl;
}



int 
main(int argc, const char **argv)
{
  parse(argc, argv);
  cout << "Loss=" << lossname 
       << " Bias=" << REGULARIZEDBIAS
       << " RegBias=" << REGULARIZEDBIAS 
       << " Lambda=" << lambda
       << endl;

  // load training set
  load(trainfile.c_str(), xtrain, ytrain);
  cout << "Number of features " << dim << "." << endl;
  int imin = 0;
  int imax = xtrain.size() - 1;
  if (trainsize > 0 && imax >= trainsize)
    imax = imin + trainsize -1;
  // prepare svm
  SvmCg svm(dim, lambda, imax-imin+1);
  Timer timer;
  // load testing set
  if (! testfile.empty())
    load(testfile.c_str(), xtest, ytest);
  int tmin = 0;
  int tmax = xtest.size() - 1;

  for(int i=0; i<epochs; i++)
    {
      cout << "--------- Epoch " << i+1 << "." << endl;
      timer.start();
      svm.train(imin, imax, xtrain, ytrain);
      timer.stop();
      cout << "Total training time " << setprecision(6) 
           << timer.elapsed() << " secs." << endl;
      svm.test(imin, imax, xtrain, ytrain, "train: ");
      if (tmax >= tmin)
        svm.test(tmin, tmax, xtest, ytest, "test:  ");
    }
}
