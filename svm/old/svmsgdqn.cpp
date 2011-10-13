// -*- C++ -*-
// SVM with Stochastic Gradient Descent and diagonal Quasi-Newton approximation
// Copyright (C) 2009- Antoine Bordes

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



#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cfloat>

using namespace std;


// custom vector functions

void 
compute_inverse_ratio_and_clip(FVector &w,
                               const SVector &x, double lambda, 
                               const FVector &wp, const FVector &wpp, 
                               double loss, double cmin=1, double cmax=100)
{
  int m = max(x.size(), w.size());
  if (w.size() < m) 
    w.resize(m);
  VFloat *d = w;
  const VFloat *s = (const VFloat*) wp;
  const VFloat *sp = (const VFloat*) wpp;
  int npairs = x.npairs();
  const SVector::Pair *pairs = x;
  int j = 0;
  double diffw=0;
  cmin = cmin * lambda;
  cmax = cmax * lambda;
  for (int i=0; i<npairs; i++, pairs++)
    {
      for (; j < pairs->i; j++)
	d[j] = lambda;
      j = pairs->i;
      diffw = s[j]-sp[j];
      if (diffw)
        {
          VFloat x = lambda + loss*pairs->v / diffw;
          if (x > cmax) x = cmax;
          if (x < cmin) x = cmin;
          d[j] = x;
        }
      else if (pairs->v)
      	d[i] = lambda + cmax;
      else
      	d[j] = lambda;
      j++;
    }
  for (; j<m; j++)
    d[j] = lambda;
}


void
compute_inverse_ratio_and_clip(FVector &w,
                               const FVector &x, double lambda, 
                               const FVector &wp, const FVector &wpp, 
                               double loss, double cmin=1, double cmax=100)
{
  int m = max(x.size(), w.size());
  if (w.size() < m) 
    w.resize(m);
  VFloat *d = w;
  const VFloat *sx = (const VFloat*) x;
  const VFloat *s = (const VFloat*) wp;
  const VFloat *sp = (const VFloat*) wpp;
  cmin = cmin * lambda;
  cmax = cmax * lambda;
  for (int i=0; i<m; i++)
    {
      double diffw = s[i]-sp[i];
      if(diffw)
        {
          VFloat x = lambda + loss*sx[i] / diffw;
          if (x > cmax) x = cmax;
          if (x < cmin) x = cmin;
          d[i] = x;
        }
      else if (sx[i])
      	d[i] = lambda + cmax;
      else
        d[i] = lambda;
    }
}




using namespace std;

typedef vector<SVector> xvec_t;
typedef vector<double> yvec_t;


// Select loss
#ifndef LOSS
# define LOSS SQUAREDHINGELOSS
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

inline 
double loss(double z)
{
#if LOSS == LOGLOSS
  if (z >= 0)
    return log(1+exp(-z));
  else
    return -z + log(1+exp(z));
#elif LOSS == LOGLOSSMARGIN
  if (z >= 1)
    return log(1+exp(1-z));
  else
    return 1-z + log(1+exp(z-1));
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
  if (z < 0)
    return 1 / (exp(z) + 1);
  double ez = exp(-z);
  return ez / (ez + 1);
#elif LOSS == LOGLOSSMARGIN
  if (z < 1)
    return 1 / (exp(z-1) + 1);
  double ez = exp(1-z);
  return ez / (ez + 1);
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





// -- stochastic gradient

class SgdQn
{
public:
  SgdQn(int dim, double lambda, double t0);
  
  double printQnInfo(const FVector &Bb, double init);
  
  void calibrate(int imin, int imax, 
		 const xvec_t &xp, const yvec_t &yp, bool verb);

  void train(int imin, int imax, 
             const xvec_t &x, const yvec_t &y,
	     const char *prefix, bool verb);

  void test(int imin, int imax, 
	    const xvec_t &x, const yvec_t &y, 
	    const char *prefix, bool verb, FVector &infos);

  FVector copy_w()
  {return w;}
  
private:
  double  t;
  double  t0;
  double  lambda;
  FVector w;
  int skip;
  int count;

  FVector B;
  FVector Bc;
  double  lastt;
};



SgdQn::SgdQn(int dim, double l, double t0)
  : t(0), t0(t0), lambda(l), w(dim), skip(1000),
    B(dim), Bc(dim), lastt(0)
{
  for(int i=0; i<dim; i++)
    Bc.set(i, 1/(lambda*t0));
}


double
SgdQn::printQnInfo(const FVector &Bb, double init)
{
  double bmax=-DBL_MAX, bmin=DBL_MAX, bmean=0.;
  int minb=0, maxb=0, notchg=0, imin=0, imax=0;
  for(int i=1; i<Bb.size();i++) // x[0] always zero
    {
      if(Bb[i]<bmin)
	bmin = Bb[i], imin=i;
      if(Bb[i]>bmax)
	bmax = Bb[i], imax=i;
      bmean+=Bb[i];      
      if(Bb[i]==init)
	notchg++;
    }
  for(int i=0; i<Bb.size();i++) // x[0] always zero
    {
      if(Bb[i]==bmax)
	maxb++;
      if(Bb[i]==bmin)
	minb++;
    }
  bmean /= (Bb.size() - 1);
  //cout << Bb << endl;
  cout  << "Bmax: " << bmax << " (i:"<<imax <<" ,@max: " 
        << maxb << "/" << Bb.size() <<  ")" << endl
        << "Bmin: " << bmin << " (i:"<<imin <<" ,@min: " 
        << minb << "/" << Bb.size()<<  ")" << endl
        << "Bmean: " << bmean << " Didnt Change: " << notchg << endl;
  return bmean;
}


void 
SgdQn::calibrate(int imin, int imax, 
		 const xvec_t &xp, const yvec_t &yp, bool verb)
{
  if (verb) 
    cout << "Estimating sparsity" << endl;
  int j;
  
  // compute average gradient size
  double n = 0;
  double s = 0;
  
  for (j=imin; j<=imax; j++,n++)
    {
      const SVector &x = xp.at(j);
      n += 1;
      s += x.npairs();
    }
  
  // compute weight decay skip
  skip = (int) ((8 * n * w.size()) / s);
  if (verb) 
    {
      cout << " using " << n << " examples." << endl;
      cout << " skip: " << skip << endl;
    }
}



void 
SgdQn::train(int imin, int imax, 
	     const xvec_t &xp, const yvec_t &yp,
	     const char *prefix, bool verb)
{
  if (verb)
    cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);

  count = skip;
  bool updateB = false;
  FVector w_1 = w;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double z = y * dot(w, x);
      double dl = dloss(z);
      t += 1;      
      if (updateB)
        {
	  double diffloss = dl - dloss(y * dot(w_1, x));
	  if (diffloss)
	    {
	      compute_inverse_ratio_and_clip(B, x, lambda, w_1, w, y*diffloss, 0.1, 100);
              VFloat *bc = Bc;
              const VFloat *b = B;
              VFloat dt = t - lastt;
              int m = Bc.size();
              for (int j=0; j<m; j++)
                bc[j] = 1.0 / ( 1.0 / bc[j] + dt * b[j] );
              lastt = t;
	    }
	  updateB=false;	
        }
      // normal update
      if(--count <= 0)
        {
          w_1 = w;
          updateB = true;
          w.add(w, -skip*lambda, Bc);
          count = skip;
        }      
#if LOSS < LOGLOSS
      if (z < 1)
#endif
        w.add(x, dl*y, Bc);       
    }

  if (verb)
    printQnInfo(Bc, 1/lambda);
  if (verb)
    cout << prefix << setprecision(6) << "Norm2: " << dot(w,w) << endl;
}



void
SgdQn::test(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix, bool verb, FVector &infos)
{
  if (verb)
    cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  int nerr = 0;
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double z = y * dot(w,x);
      if (z <= 0)
        nerr += 1;
#if LOSS < LOGLOSS
      if (z < 1)
#endif
        cost += loss(z);
    }
  int n = imax - imin + 1;
  double loss = cost / n;
  cost = loss + 0.5 * lambda * dot(w,w);

  if (verb)
    {
      cout << prefix << setprecision(4)
	   << "Misclassification: " << (double)nerr * 100.0 / n << "%." << endl;
      cout << prefix << setprecision(12) 
	   << "Cost: " << cost << "." << endl;
      cout << prefix << setprecision(12) 
	   << "Loss: " << loss << "." << endl;
    }
  infos[1]=(double)nerr/n;
  infos[2]=cost;
}



// --- options

string trainfile;
string testfile;
string logfile;

double lambda = 1e-4;
double t0 = 0;
int epochs = 5;
int trainsize = -1;
double steps=1.;

void 
usage()
{
  cerr << "Usage: (sparse_)svmsgdqn [options] trainfile [testfile] [logfile]" << endl
       << "Options:" << endl
       << " -lambda <lambda> (default 1e-4)" << endl
       << " -epochs <epochs> (default 5)" << endl
       << " -t0 <t0>: if none given (default), an automatic procedure selects one." << endl
       << " -trainsize <n>" << endl
       << " -steps <s>: number of intermediate values to be printed in the logfile (including a test error)."  << endl   
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
          else if (logfile.empty())
            logfile = arg;
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
          else if (opt == "t0" && i+1<argc)
            {
              t0 = atof(argv[++i]);
              cout << "Using t0=" << t0 << "." << endl;
	      assert(t0>0);
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
          else if (opt == "steps" && i+1<argc)
            {
              steps = atoi(argv[++i]);
              assert(steps > 0);
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
ofstream logs;

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

double determine_t0(int imin, int imax, int epochs)
{  

  cout << "Estimating t0 ..." << endl;
  double t0 = 1;
  double t0tmp = 1;
  double lowest_cost=DBL_MAX;
  for (int i=0; i<=10; i++)
    {
      SgdQn svm(dim, lambda, t0tmp);
      svm.calibrate(imin, (int)imax/10, xtrain, ytrain, false);
      for (int ep=0; ep<epochs; ep++)
	svm.train(imin, (int)imax/10, xtrain, ytrain, "train: ", false);
      FVector info(2);
      svm.test(imin, (int)imax/10, xtrain, ytrain, "train: ", false, info);
      double cost= info[2];
      if (cost<lowest_cost && cost==cost) // check for NaN
	{
	  t0=t0tmp;
	  lowest_cost=cost;
	}      
      cout  << " t0=" << t0tmp << ", cost="<<cost <<endl;
      t0tmp=t0tmp*10;
    }
  cout <<   "Final choice: t0=" << t0 << endl;	 
  return t0;
}

int 
main(int argc, const char **argv)
{
  parse(argc, argv);
  cout << "Loss=" << lossname 
       << " Bias=" << 0 
       << " RegBias=" << 0
       << " Lambda=" << lambda
       << endl;

  // load training set
  load(trainfile.c_str(), xtrain, ytrain);
  cout << "Number of features " << dim << "." << endl;
  int imin = 0;
  int imax = xtrain.size() - 1;
  if (trainsize > 0 && imax >= trainsize)
    imax = imin + trainsize -1;

  // load testing set
  if (! testfile.empty())
    {
      load(testfile.c_str(), xtest, ytest);
    }
  int tmin = 0;
  int tmax = xtest.size() - 1;
  
  // prepare svm
  if(t0==0)
    t0 = determine_t0(imin, imax, epochs);

  SgdQn svm(dim, lambda, t0);
  svm.calibrate(0, imax, xtrain, ytrain, true);

  if (! logfile.empty())
    {
      logs.open(logfile.c_str());
      logs << "# SVMSGDQN. lambda = " << lambda << " , t0 = " << t0 << endl;
      logs << "# trnsz = " << imax-imin+1 << " , tstsz = " << tmax-tmin+1 << endl;
      logs << "# it time trn_err trn_cost tst_err tst_cost" << endl;
      FVector info(2);
      svm.test(imin, imax, xtrain, ytrain, "train: ", true, info);
      logs << 0 << " " << 0 << " " << info[1] << " " << info[2];
      if (tmax >= tmin)
	{
	  svm.test(tmin, tmax, xtest, ytest, "test:  ", true, info);
	  if (! logfile.empty())
	    logs << " " << info[1] << " " << info[2];
	}
      logs << endl;
    }

  Timer timer;
  if(! logfile.empty())
    for(int i=0; i<epochs; i++)
      {
	FVector info(2);
	cout << "--------- Epoch " << i+1 << "." << endl;
	for (int j=0;j<(int)steps;j++)
	  {
	    int idxmin=imin+(int)((double)imax*j/steps);
	    int idxmax=imin+(int)((double)imax*(j+1)/steps);
	    timer.start();
	    svm.train(idxmin, idxmax, xtrain, ytrain, "train: ",true);
	    timer.stop();
	    cout << "Total training time " << setprecision(6) 
		 << timer.elapsed() << " secs." << endl;
	    svm.test(imin, imax, xtrain, ytrain, "train: ", true, info);
	    logs << i+(double)(j+1)/steps << " " << timer.elapsed() << " " << info[1] << " " << info[2];
	    if (tmax >= tmin)
	      {
		svm.test(tmin, tmax, xtest, ytest, "test:  ", true, info);
		logs << " " << info[1] << " " << info[2];
	      }
	    logs << endl;
	  }
      }
  else
    for(int i=0; i<epochs; i++)
      {
	FVector info(2);
	cout << "--------- Epoch " << i+1 << "." << endl;
	timer.start();
	svm.train(imin, imax, xtrain, ytrain, "train: ", true);
	timer.stop();
	cout << "Total training time " << setprecision(6)
	     << timer.elapsed() << " secs." << endl;
	svm.test(imin, imax, xtrain, ytrain, "train: ", true, info);
	if (tmax >= tmin)
	    svm.test(tmin, tmax, xtest, ytest, "test:  ", true, info);
      }
}
