// -*- C++ -*-
// CRF with stochastic gradient
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


#include "wrapper.h"
#include "vectors.h"
#include "matrices.h"
#include "gzstream.h"
#include "pstream.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cctype>
#include <cmath>

using namespace std;

#if defined(_GXX_EXPERIMENTAL_CXX0X__)
# include <unordered_map>
# define hash_map unordered_map
#elsif defined(__GNUC__)
# include <ext/hash_map>
using __gnu_cxx::hash_map;
namespace __gnu_cxx {
  template<>
  struct hash<string> {
    hash<char*> h;
    inline size_t operator()(const string &s) const { return h(s.c_str());
    };
  };
};
#else
# define hash_map map
#endif

#ifndef HUGE_VAL
# define HUGE_VAL 1e+100
#endif

typedef vector<string> strings_t;
typedef vector<int> ints_t;

bool verbose = true;


// ============================================================
// Utilities


static int
skipBlank(istream &f)
{
  int c = f.get();
  while (f.good() && isspace(c) && c!='\n' && c!='\r')
    c = f.get();
  f.unget();  
  return c;
}


static int
skipSpace(istream &f)
{
  int c = f.get();
  while (f.good() && isspace(c))
    c = f.get();
  f.unget();  
  return c;
}


inline double
expmx(double x)
{
#define EXACT_EXPONENTIAL 0
#if EXACT_EXPONENTIAL
  return exp(-x);
#else
  // fast approximation of exp(-x) for x positive
# define A0   (1.0)
# define A1   (0.125)
# define A2   (0.0078125)
# define A3   (0.00032552083)
# define A4   (1.0172526e-5) 
  if (x < 13.0) 
    {
      assert(x>=0);
      double y;
      y = A0+x*(A1+x*(A2+x*(A3+x*A4)));
      y *= y;
      y *= y;
      y *= y;
      y = 1/y;
      return y;
    }
  return 0;
# undef A0
# undef A1
# undef A2
# undef A3
# undef A4
#endif
}


static double
logSum(const VFloat *v, int n)
{
  int i;
  VFloat m = v[0];
  for (i=0; i<n; i++)
    m = max(m, v[i]);
  double s = 0;
  for (i=0; i<n; i++)
    s += expmx(m-v[i]);
  return m + log(s);
}


static double
logSum(const FVector &v)
{
  return logSum(v, v.size());
}


static void
dLogSum(double g, const VFloat *v, VFloat *r, int n)
{
  int i;
  VFloat m = v[0];
  for (i=0; i<n; i++)
    m = max(m, v[i]);
  double z = 0;
  for (i=0; i<n; i++)
    {
      double e = expmx(m-v[i]);
      r[i] = e;
      z += e;
    }
  for (i=0; i<n; i++)
    r[i] = g * r[i] / z;
}


static void
dLogSum(double g, const FVector &v, FVector &r)
{
  assert(v.size() <= r.size());
  dLogSum(g, v, r, v.size());
}


class ixstream_t
{
  bool z;
  ifstream fn;
  igzstream fz;

public:
  ixstream_t(const char *name) 
  {
    string fname = name;
    int len = fname.size();
    z = fname.substr(len-3) == ".gz";
    if (z)
      fz.open(name);
    else
      fn.open(name);
  }
  istream& stream()
  {
    if (z)
      return fz;
    else
      return fn;
  }
};


// ============================================================
// Parsing data file


int
readDataLine(istream &f, strings_t &line, int &expected)
{
  int obtained = 0;
  while (f.good())
    {
      int c = skipBlank(f);
      if (c == '\n' || c == '\r')
        break;
      string s;
      f >> s;
      if (! s.empty())
        {
          line.push_back(s);
          obtained += 1;
        }
    }
  int c = f.get();
  if (c == '\r' && f.get() != '\n')
    f.unget();
  if (obtained > 0)
    {
      if (expected <= 0)
        expected = obtained;
      else if (expected > 0 && expected != obtained)
        {
          cerr << "ERROR: expecting " << expected 
               << " columns in data file." << endl;
          exit(10);
        }
    }
  else
    skipSpace(f);
  return obtained;
}


int 
readDataSentence(istream &f, strings_t &s, int &expected)
{
  strings_t line;
  s.clear();
  while (f.good())
    if (readDataLine(f, s, expected))
      break;
  while (f.good())
    if (! readDataLine(f, s, expected))
      break;
  if (expected)
    return s.size() / expected;
  return 0;
}



// ============================================================
// Processing templates


void
checkTemplate(string tpl)
{
  const char *p = tpl.c_str();
  if (p[0]!='U' && p[0]!='B')
    {
      cerr << "ERROR: Unrecognized template type (neither U nor B.)" << endl
           << "       Template was \"" << tpl << "\"." << endl;
      exit(10);
    }
  while (p[0])
    {
      if (p[0]=='%' && p[1]=='x')
        {
          bool okay = false;
          char *n = const_cast<char*>(p);
          long junk;
          if (n[2]=='[') {
            junk = strtol(n+3,&n, 10);
            while (isspace(n[0]))
              n += 1;
            if (n[0] == ',') {
              junk = strtol(n+1, &n, 10);
              while (isspace(n[0]))
                n += 1;
              if (n[0] == ']')
                okay = true;
            }
          }
          if (okay)
            p = n;
          else {
            cerr << "ERROR: Syntax error in %x[.,,] expression." << endl
                 << "       Template was \"" << tpl << "\"." << endl;
            exit(10);
          }
        }
      p += 1;
    }
}


string
expandTemplate(string tpl, const strings_t &s, int columns, int pos)
{
  string e;
  int rows = s.size() / columns;
  const char *t = tpl.c_str();
  const char *p = t;
  
  static const char *BOS[4] = { "_B-1", "_B-2", "_B-3", "_B-4"};
  static const char *EOS[4] = { "_B+1", "_B+2", "_B+3", "_B+4"};

  while (*p)
    {
      if (p[0]=='%' && p[1]=='x' && p[2]=='[')
        {
          if (p > t)
            e.append(t, p-t);
          // parse %x[A,B] assuming syntax has been verified
          char *n;
          int a = strtol(p+3, &n, 10);
          while (n[0] && n[0]!=',')
            n += 1;
          int b = strtol(n+1, &n, 10);
          while (n[0] && n[0]!=']')
            n += 1;
          p = n;
          t = n+1;
          // catenate
          a += pos;
          if (b>=0 && b<columns)
            {
              if (a>=0 && a<rows)
                e.append(s[a*columns+b]);
              else if (a<0)
                e.append(BOS[min(3,-a-1)]);
              else if (a>=rows)
                e.append(EOS[min(3,a-rows)]);
            }
        }
      p += 1;
    }
  if (p > t)
    e.append(t, p-t);
  return e;
}


void
readTemplateFile(const char *fname, strings_t &templateVector)
{
  ifstream f(fname);
  if (! f.good())
    {
      cerr << "ERROR: Cannot open " << fname << " for reading." << endl;
      exit(10);
    }
  while(f.good())
    {
      int c = skipSpace(f);
      while (c == '#')
        {
          while (f.good() && c!='\n' && c!='\r')
            c = f.get();
          f.unget();
          c = skipSpace(f);
        }
      string s;
      getline(f,s);
      if (! s.empty())
        {
          checkTemplate(s);
          templateVector.push_back(s);        
        }
    }
  if (! f.eof())
    {
      cerr << "ERROR: Cannot read " << fname << " for reading." << endl;
      exit(10);
    }
}



// ============================================================
// Dictionary


typedef hash_map<string,int> dict_t;

class Dictionary
{
private:
  dict_t outputs;
  dict_t features;
  strings_t templates;
  strings_t outputnames;
  mutable dict_t internedStrings;
  int index;

public:
  Dictionary() : index(0) { }

  int nOutputs() const { return outputs.size(); }
  int nFeatures() const { return features.size(); }
  int nTemplates() const { return templates.size(); }
  int nParams() const { return index; }
  
  int output(string s) const { 
    dict_t::const_iterator it = outputs.find(s);
    return (it != outputs.end()) ? it->second : -1;
  }
  
  int feature(string s) const { 
    dict_t::const_iterator it = features.find(s);
    return (it != features.end()) ? it->second : -1;
  }

  string outputString(int i) const { return outputnames.at(i); }
  string templateString(int i) const { return templates.at(i); }

  string internString(string s) const;
  
  int initFromData(const char *tFile, const char *dFile, int cutoff=1);

  friend istream& operator>> ( istream &f, Dictionary &d );
  friend ostream& operator<< ( ostream &f, const Dictionary &d );
};



string
Dictionary::internString(string s) const
{
  dict_t::const_iterator it = internedStrings.find(s);
  if (it != internedStrings.end())
    return it->first;
#if defined(mutable)
  const_cast<Dictionary*>(this)->
#endif
  internedStrings[s] = 1;
  return s;
}


ostream&
operator<<(ostream &f, const Dictionary &d)
{
  typedef map<int,string> rev_t;
  rev_t rev;
  strings_t::const_iterator si;
  dict_t::const_iterator di;
  rev_t::const_iterator ri;
  for (di=d.outputs.begin(); di!=d.outputs.end(); di++)
    rev[di->second] = di->first;
  for (ri=rev.begin(); ri!=rev.end(); ri++)
    f << "Y" << ri->second << endl;
  for (si=d.templates.begin(); si!=d.templates.end(); si++)
    f << "T" << *si << endl;
  rev.clear();
  for (di=d.features.begin(); di!=d.features.end(); di++)
    rev[di->second] = di->first;
  for (ri=rev.begin(); ri!=rev.end(); ri++)
    f << "X" << ri->second << endl;
  return f;
}


istream& 
operator>>(istream &f, Dictionary &d)
{
  d.outputs.clear();
  d.features.clear();
  d.templates.clear();
  d.index = 0;
  int findex = 0;
  int oindex = 0;
  while (f.good())
    {
      string v;
      skipSpace(f);
      int c = f.get();
      if  (c == 'Y')
        {
          f >> v;
          if (v.empty())
            {
              cerr << "ERROR (reading dictionary): " 
                   << "Invalid Y record in model file." << endl;
              exit(10);
            }
          if (findex>0)
            {
              cerr << "ERROR (reading dictionary): " 
                   << "Found Y record occuring after X record." << endl;
              exit(10);
            }
          d.outputs[v] = oindex++;
        }
      else if (c == 'T')
        {
          f >> v;
          if (v.empty())
            {
              cerr << "ERROR (reading dictionary): " 
                   << "Invalid T record." << endl;
              exit(10);
            }
          checkTemplate(v);
          d.templates.push_back(v);
        }
      else if (c == 'X')
        {
          f >> v;
          if (v.empty())
            {
              cerr << "ERROR (reading dictionary): " 
                   << "Invalid X record." << endl;
              exit(10);
            }
          int nindex = findex;
          if (v[0]=='U')
            nindex += oindex;
          else if (v[0]=='B')
            nindex += oindex * oindex;
          else
            {
              cerr << "ERROR (reading dictionary): " 
                   << "Invalid feature in X record: " << v << endl;
              exit(10);
            }
          d.features[v] = findex;
          findex = nindex;
        }
      else
        {
          f.unget();
          break;
        }
    }
  d.index = findex;
  if (!f.good() && !f.eof())
    {
      d.outputs.clear();
      d.features.clear();
      d.templates.clear();
      d.index = 0;
    }
  d.outputnames.resize(oindex);
  for (dict_t::const_iterator it=d.outputs.begin(); 
       it!=d.outputs.end(); it++)
    d.outputnames[it->second] = it->first;
  return f;
}



typedef pair<string,int> sipair_t;
typedef vector<sipair_t> sivector_t;

struct SIPairCompare {
  bool operator() (const sipair_t &p1, const sipair_t &p2) {
    if (p1.second > p2.second) return true;
    else if (p1.second < p2.second) return false;
    else return (p1.first < p2.first);
  }
} siPairCompare;

int
Dictionary::initFromData(const char *tFile, const char *dFile, int cutoff)
{
  // clear all
  templates.clear();
  outputs.clear();
  features.clear();
  index = 0;
  
  // read templates
  if (verbose)
    cout << "Reading template file " << tFile << "." << endl;
  readTemplateFile(tFile, templates);
  int nu = 0;
  int nb = 0;
  for (unsigned int t=0; t<templates.size(); t++)
    if (templates[t][0]=='U')
      nu += 1;
    else if (templates[t][0]=='B')
      nb += 1;
  if (verbose)
    cout << "  u-templates: " << nu 
         << "  b-templates: " << nb << endl;
  if (nu + nb != (int)templates.size())
    {
      cerr << "ERROR (building dictionary): "
           << "Problem counting templates" << endl;
      exit(10);
    }
  
  // process compressed datafile
  if (verbose)
    cerr << "Scanning " << dFile << " to build dictionary." << endl;
  typedef hash_map<string,int> hash_t;
  hash_t fcount;
  int columns = 0;
  int oindex = 0;
  int sentences = 0;
  strings_t s;
  ixstream_t fx(dFile);
  istream &f = fx.stream();
  Timer timer;
  timer.start();
  while (readDataSentence(f, s, columns))
    {
      sentences += 1;
      // intern strings to save memory
      for (strings_t::iterator it=s.begin(); it!=s.end(); it++)
        *it = internString(*it);
      // expand features and count them
      int rows = s.size()/columns;
      for (int pos=0; pos<rows; pos++)
        {
          // check output keyword
          string &y = s[pos*columns+columns-1];
          dict_t::iterator di = outputs.find(y);
          if (di == outputs.end())
            outputs[y] = oindex++;
          // expand templates
          for (unsigned int t=0; t<templates.size(); t++)
            {
              string x = expandTemplate(templates[t], s, columns, pos);
              hash_t::iterator hi = fcount.find(x);
              if (hi != fcount.end())
                hi->second += 1;
              else
                fcount[x] = 1;
            }
        }
    }
  if (! f.eof())
    {
      cerr << "ERROR (building dictionary): "
           << "Problem reading data file " << dFile << endl;
      exit(10);
    }
  outputnames.resize(oindex);
  for (dict_t::const_iterator it=outputs.begin(); it!=outputs.end(); it++)
    outputnames[it->second] = it->first;
  if (verbose)
    cout << "  sentences: " << sentences 
         << "  outputs: " << oindex << endl;
  
  // sorting in frequency order
  sivector_t keys;
  for (hash_t::iterator hi = fcount.begin(); hi != fcount.end(); hi++)
    if (hi->second >= cutoff)
      keys.push_back(*hi);
  if (keys.size() <= 0)
    {
      cerr << "ERROR (building dictionary): "
           << "No features satisfy the cutoff frequency" << endl;
      exit(10);
    }
  sort(keys.begin(), keys.end(), siPairCompare);

  // allocating parameters
  for (unsigned int j=0; j<keys.size(); j++)
    {
      string k = keys[j].first;
      features[k] = index;
      if (k[0] == 'B')
        index += oindex * oindex;
      else
        index += oindex;
    }
  if (verbose)
    cout << "  cutoff: " << cutoff 
         << "  features: " << features.size() 
         << "  parameters: " << index << endl
         << "  duration: " << timer.elapsed() << " seconds." << endl;

  return sentences;
}



// ============================================================
// Preprocessing data


typedef vector<SVector> svec_t;
typedef vector<int> ivec_t;


class Sentence
{
private:
  struct Rep 
  {
    int refcount;
    int columns;
    strings_t data;
    svec_t uFeatures;
    svec_t bFeatures;
    ivec_t yLabels;
    Rep *copy() { return new Rep(*this); }
  };
  Wrapper<Rep> w;
  Rep *rep() { return w.rep(); }
  const Rep *rep() const { return w.rep(); }

public:
  Sentence() {}

  void init(const Dictionary &dict, const strings_t &s, int columns);

  int size() const { return rep()->uFeatures.size(); }
  SVector u(int i) const { return rep()->uFeatures.at(i); }
  SVector b(int i) const { return rep()->bFeatures.at(i); }
  int y(int i) const { return rep()->yLabels.at(i); }
  
  int columns() const { return rep()->columns; }
  string data(int pos, int col) const;

  friend ostream& operator<<(ostream &f, const Sentence &s);
};


void
Sentence::init(const Dictionary &dict, const strings_t &s, int columns)
{
  w.detach();
  Rep *r = rep();
  int maxcol = columns - 1;
  int maxpos = s.size()/columns - 1;
  int ntemplat = dict.nTemplates();
  r->uFeatures.clear();
  r->bFeatures.clear();
  r->yLabels.clear();
  r->columns = columns;
  // intern strings to save memory
  for (strings_t::const_iterator it=s.begin(); it!=s.end(); it++)
    r->data.push_back(dict.internString(*it));
  // expand features
  for (int pos=0; pos<=maxpos; pos++)
    {
      // labels
      string y = s[pos*columns+maxcol];
      int yindex = dict.output(y);
      r->yLabels.push_back(yindex);
      // features
      SVector u;
      SVector b;
      for (int t=0; t<ntemplat; t++)
        {
          string tpl = dict.templateString(t); 
          int findex = dict.feature(expandTemplate(tpl, s, columns, pos));
          if (findex >= 0)
            {
              if (tpl[0]=='U')
                u.set(findex, 1);
              else if (tpl[0]=='B')
                b.set(findex, 1);
            }
        }
      r->uFeatures.push_back(u);
      if (pos < maxpos)
        r->bFeatures.push_back(b);
    }
}


string
Sentence::data(int pos, int col) const
{
  const Rep *r = rep();
  if (pos>=0 && pos<size())
    if (col>=0 && col<r->columns)
      return r->data[pos*r->columns+col];
  return string();
}


ostream&
operator<<(ostream &f, const Sentence &s)
{
  int maxpos = s.size() - 1;
  int columns = s.columns();
  for (int pos = 0; pos<=maxpos; pos++) {
    for (int col = 0; col<columns; col++)
      f << s.data(pos, col) << " ";
    f << endl << "   Y" << pos << " " << s.y(pos) << endl;
    f << "   U" << pos << s.u(pos);
    if (pos < maxpos)
      f << "   B" << pos << s.b(pos);
  }
  return f;
}


typedef vector<Sentence> dataset_t;


void
loadSentences(const char *fname, const Dictionary &dict, dataset_t &data)
{
  if (verbose)
    cout << "Reading and preprocessing " << fname << "." << endl;
  Timer timer;
  int sentences = 0;
  int columns = 0;
  strings_t s;
  ixstream_t fx(fname);
  istream &f = fx.stream();
  timer.start();
  while (readDataSentence(f, s, columns))
    {
      Sentence ps;
      ps.init(dict, s, columns);
      data.push_back(ps);
      sentences += 1;
    }
  if (verbose)
    cout << "  processed: " << sentences << " sentences." << endl
         << "  duration: " << timer.elapsed() << " seconds." << endl;
}




// ============================================================
// Scorer


class Scorer
{
public:
  Sentence s;
  const Dictionary &d;
  VFloat *w;
  double &wscale;
  bool scoresOk;
  vector<FVector> uScores;
  vector<FMatrix> bScores;

  Scorer(const Sentence &s_, const Dictionary &d_, FVector &w_, double &c_);
  void computeScores();
  virtual ~Scorer() {}
  virtual void uGradients(const VFloat *g, int pos, int fy, int ny) {}
  virtual void bGradients(const VFloat *g, int pos, int fy, int ny, int y) {}
  
  double viterbi(ints_t &path);
  int test();
  int test(ostream &f);
  double scoreCorrect();
  double gradCorrect(double g);
  double scoreForward();
  double gradForward(double g);
};


Scorer::Scorer(const Sentence &s_, const Dictionary &d_, FVector &w_, double &c_)
  : s(s_), d(d_), w(w_), wscale(c_), scoresOk(false)
{
  assert(w_.size() == d.nParams());
}

void
Scorer::computeScores()
{
  if (! scoresOk)
    {
      int nout = d.nOutputs();
      int npos = s.size();
      int pos;
      // compute uScores
      uScores.resize(npos);
      for (pos=0; pos<npos; pos++)
        {
          FVector &u = uScores[pos];
          u.resize(nout);
          SVector x = s.u(pos);
          for (const SVector::Pair *p = x; p->i >= 0; p++)
            for (int j=0; j<nout; j++)
              u[j] += w[p->i + j] * p->v;
          for (int j=0; j<nout; j++)
            u[j] *= wscale;
        }
      // compute bScores
      bScores.resize(npos-1);
      for (pos=0; pos<npos-1; pos++)
        {
          FMatrix &b = bScores[pos];
          b.resize(nout,nout);
          SVector x = s.b(pos);
          for (const SVector::Pair *p = x; p->i >= 0; p++)
            { 
              int k = 0;
              for (int i=0; i<nout; i++)
                {
                  FVector &bi = b[i];
                  for (int j=0; j<nout; j++, k++)
                    bi[j] += w[p->i + k] * p->v;
                }
            }
          for (int i=0; i<nout; i++) 
            {
              FVector &bi = b[i];
              for (int j=0; j<nout; j++)
                bi[j] *= wscale;
            }
        }
    }
  scoresOk = true;
}


double 
Scorer::viterbi(ints_t &path)
{
  computeScores();
  int npos = s.size();
  int nout = d.nOutputs();
  int pos, i, j;
  
  // allocate backpointer array
  vector<ints_t> pointers(npos);
  for (int i=0; i<npos; i++)
    pointers[i].resize(nout);
  
  // process scores
  FVector scores = uScores[0];
  for (pos=1; pos<npos; pos++)
    {
      FVector us = uScores[pos];
      for (i=0; i<nout; i++)
        {
          FVector bs = bScores[pos-1][i];
          bs.add(scores);
          int bestj = 0;
          double bests = bs[0];
          for (j=1; j<nout; j++)
            if (bs[j] > bests)
              { bests = bs[j]; bestj = j; }
          pointers[pos][i] = bestj;
          us[i] += bests;
        }
      scores = us;
    }
  // find best final score
  int bestj = 0;
  double bests = scores[0];
  for (j=1; j<nout; j++)
    if (scores[j] > bests)
      { bests = scores[j]; bestj = j; }
  // backtrack
  path.resize(npos);
  for (pos = npos-1; pos>=0; pos--)
    {
      path[pos] = bestj;
      bestj = pointers[pos][bestj];
    }
  return bests;
}


int
Scorer::test()
{
  ints_t path;
  int npos = s.size();
  int errors = 0;
  viterbi(path);
  for (int pos=0; pos<npos; pos++)
    if (path[pos] != s.y(pos))
      errors += 1;
  return errors;
}

int
Scorer::test(ostream &f)
{
  ints_t path;
  int npos = s.size();
  int ncol = s.columns();
  int errors = 0;
  viterbi(path);
  for (int pos=0; pos<npos; pos++)
    {
      if (path[pos] != s.y(pos))
        errors += 1;
      for (int c=0; c<ncol; c++)
        f << s.data(pos,c) << " ";
      f << d.outputString(path[pos]) << endl;
    }
  f << endl;
  return errors;
}


double 
Scorer::scoreCorrect()
{
  computeScores();
  int npos = s.size();
  int y = s.y(0);
  double sum = uScores[0][y];
  for (int pos=1; pos<npos; pos++)
    {
      int fy = y;
      y = s.y(pos);
      if (y>=0 && fy>=0)
        sum += bScores[pos-1][y][fy];
      if (y>=0)
        sum += uScores[pos][y];
    }
  return sum;
}


double
Scorer::gradCorrect(double g)
{
  computeScores();
  int npos = s.size();
  int y = s.y(0);
  VFloat vf = g;
  uGradients(&vf, 0, y, 1);
  double sum = uScores[0][y];
  for (int pos=1; pos<npos; pos++)
    {
      int fy = y;
      y = s.y(pos);
      if (y>=0 && fy>=0)
        sum += bScores[pos-1][y][fy];
      if (y>=0)
        sum += uScores[pos][y];
      if (y>=0 && fy>=0)
        bGradients(&vf, pos-1, fy, 1, y);
      if (y>=0)
      uGradients(&vf, pos, y, 1);
    }
  return sum;
}


double 
Scorer::scoreForward()
{
  computeScores();
  int npos = s.size();
  int nout = d.nOutputs();
  int pos, i;

  FVector scores = uScores[0];
  for (pos=1; pos<npos; pos++)
    {
      FVector us = uScores[pos];
      for (i=0; i<nout; i++)
        {
          FVector bs = bScores[pos-1][i];
          bs.add(scores);
          us[i] += logSum(bs);
        }
      scores = us;
    }
  return logSum(scores);
}



double 
Scorer::gradForward(double g)
{
  computeScores();
  int npos = s.size();
  int nout = d.nOutputs();
  int pos;

#define USE_FORWARD_BACKWARD 0
#if USE_FORWARD_BACKWARD
  
  FMatrix alpha(npos,nout);
  FMatrix beta(npos,nout);
  // forward
  alpha[0] = uScores[0];
  for (pos=1; pos<npos; pos++)
    {
      FVector us = uScores[pos];
      for (int i=0; i<nout; i++)
        {
          FVector bs = bScores[pos-1][i];
          bs.add(alpha[pos-1]);
          us[i] += logSum(bs);
        }
      alpha[pos] = us;
    }

  // backward
  beta[pos-1] = uScores[pos-1];
  for (pos=pos-2; pos>=0; pos--)
    {
      FVector us = uScores[pos];
      for (int i=0; i<nout; i++)
        {
          FVector bs(nout);
          const FMatrix &bsc = bScores[pos];
          for (int j=0; j<nout; j++)
            bs[j] = bsc[j][i];
          bs.add(beta[pos+1]);
          us[i] += logSum(bs);
        }
      beta[pos] = us;
    }
  // score
  double score = logSum(beta[0]);

  // collect gradients
  for (pos=0; pos<npos; pos++)
    {
      FVector b = beta[pos];
      if (pos > 0)
        {
          FVector a = alpha[pos-1];
          const FMatrix &bsc = bScores[pos-1];
          for (int j=0; j<nout; j++)
            {
              FVector bs = bsc[j];
              FVector bgrad(nout);
              for (int i=0; i<nout; i++)
                bgrad[i] = g * expmx(max(0.0, score - bs[i] - a[i] - b[j]));
              bGradients(bgrad, pos-1, 0, nout, j);
            }
        }
      FVector a = alpha[pos];
      FVector us = uScores[pos];
      FVector ugrad(nout);
      for (int i=0; i<nout; i++)
        ugrad[i] = g * expmx(max(0.0, score + us[i] - a[i] - b[i]));
      uGradients(ugrad, pos, 0, nout);
    }

#else

  // forward
  FMatrix scores(npos, nout);
  scores[0] = uScores[0];
  for (pos=1; pos<npos; pos++)
    {
      FVector us = uScores[pos];
      for (int i=0; i<nout; i++)
        {
          FVector bs = bScores[pos-1][i];
          bs.add(scores[pos-1]);
          us[i] += logSum(bs);
        }
      scores[pos] = us;
    }
  double score = logSum(scores[npos-1]);

  // backward with chain rule
  FVector tmp(nout);
  FVector grads(nout);
  dLogSum(g, scores[npos-1], grads);
  for (pos=npos-1; pos>0; pos--)
    {
      FVector ug;
      uGradients(grads, pos, 0, nout);
      for (int i=0; i<nout; i++)
        if (grads[i])
          { 
            FVector bs = bScores[pos-1][i];
            bs.add(scores[pos-1]);
            dLogSum(grads[i], bs, tmp);
            bGradients(tmp, pos-1, 0, nout, i);
            ug.add(tmp);
          }
      grads = ug;
    }
  uGradients(grads, 0, 0, nout);

#endif

  return score;
}



// ============================================================
// GScorer - compute gradients as SVectors


class GScorer : public Scorer
{
private:
  SVector grad;
public:
  GScorer(const Sentence &s_, const Dictionary &d_, FVector &w_, double &c_);
  void clear() { grad.clear(); }
  SVector gradient() { return grad; }
  virtual void uGradients(const VFloat *g, int pos, int fy, int ny);
  virtual void bGradients(const VFloat *g, int pos, int fy, int ny, int y);
};


GScorer::GScorer(const Sentence &s_,const Dictionary &d_, 
                 FVector &w_, double &c_)
  : Scorer(s_, d_, w_, c_)
{
}


void 
GScorer::uGradients(const VFloat *g, int pos, int fy, int ny)
{
  int n = d.nOutputs();
  assert(pos>=0 && pos<s.size());
  assert(fy>=0 && fy<n);
  assert(fy+ny>0 && fy+ny<=n);
  int off = fy;
  SVector x = s.u(pos);
  SVector a;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    for (int j=0; j<ny; j++)
      a.set(p->i + off + j, g[j] * p->v);
  grad.add(a);
}


void 
GScorer::bGradients(const VFloat *g, int pos, int fy, int ny, int y)
{
  int n = d.nOutputs();
  assert(pos>=0 && pos<s.size());
  assert(y>=0 && y<n);
  assert(fy>=0 && fy<n);
  assert(fy+ny>0 && fy+ny<=n);
  int off = y * n + fy;
  SVector x = s.b(pos);
  SVector a;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    for (int j=0; j<ny; j++)
      a.set(p->i + off + j, g[j] * p->v);
  grad.add(a);
}



// ============================================================
// TScorer - training score: update weights directly


class TScorer : public Scorer
{
private:
  double eta;
public:
  TScorer(const Sentence &s_, const Dictionary &d_,
          FVector &w_, double &c_, double eta_);
  virtual void uGradients(const VFloat *g, int pos, int fy, int ny);
  virtual void bGradients(const VFloat *g, int pos, int fy, int ny, int y);
};


TScorer::TScorer(const Sentence &s_, const Dictionary &d_,
                 FVector &w_, double &c_, double eta_ )
  : Scorer(s_,d_,w_,c_), eta(eta_)
{
}


void 
TScorer::uGradients(const VFloat *g, int pos, int fy, int ny)
{
  int n = d.nOutputs();
  assert(pos>=0 && pos<s.size());
  assert(fy>=0 && fy<n);
  assert(fy+ny>0 && fy+ny<=n);
  int off = fy;
  SVector x = s.u(pos);
  double gain = eta / wscale;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    for (int j=0; j<ny; j++)
      w[p->i + off + j] += g[j] * p->v * gain;
}


void 
TScorer::bGradients(const VFloat *g, int pos, int fy, int ny, int y)
{
  int n = d.nOutputs();
  assert(pos>=0 && pos<s.size());
  assert(y>=0 && y<n);
  assert(fy>=0 && fy<n);
  assert(fy+ny>0 && fy+ny<=n);
  int off = y * n + fy;
  SVector x = s.b(pos);
  SVector a;
  double gain = eta / wscale;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    for (int j=0; j<ny; j++)
      w[p->i + off + j] += g[j] * p->v * gain;
}




// ============================================================
// Main class CrfSgd




class CrfSgd
{
  Dictionary dict;
  double wscale;
  FVector w;
  double lambda;
  double wnorm;
  double t;
  int epoch;

  void load(istream &f);
  void save(ostream &f) const;
  void rescale();
  
  double findObjBySampling(const dataset_t &data, const ivec_t &sample);
  double tryEtaBySampling(const dataset_t &data, const ivec_t &sample, 
                          double eta);
  
public:
  
  CrfSgd();
  
  int getEpoch() const { return epoch; }
  const Dictionary& getDict() const { return dict; }
  double getLambda() const { return lambda; }
  double getEta() const { return 1/(t*lambda); }
  FVector getW() const { const_cast<CrfSgd*>(this)->rescale(); return w; }
  

  void initialize(const char *templatefile, 
                  const char *datafile, 
                  double c = 4,
                  int cutoff = 3);
  
  void adjustEta(double seta=1);

  void adjustEta(const dataset_t &data, int sample=500, 
                 double seta=1, Timer *tm=0);

  void train(const dataset_t &data, int epochs=1, Timer *tm=0);

  void test(const dataset_t &data, const char *conlleval=0, Timer *tm=0);
  
  friend istream& operator>> ( istream &f, CrfSgd &d );
  friend ostream& operator<< ( ostream &f, const CrfSgd &d );
};


CrfSgd::CrfSgd()
  : wscale(1), lambda(0), wnorm(0), t(0), epoch(0)
{
}

void
CrfSgd::load(istream &f)
{
  f >> dict;
  t = 0;
  epoch = 0;
  wscale = 0;
  w.clear();
  
  while (f.good())
    {
      skipSpace(f);
      int c = f.get();
      if (f.eof())
        break;
      if (c == 'T')
        {
          t = -1;
          f >> t;
          if (!f.good() || t <= 0)
            {
              cerr << "ERROR (reading model): "
                   << "Invalid iteration number: " << t << endl;
              exit(10);
            }
        }
      else if (c == 'E')
        {
          epoch = -1;
          f >> epoch;
          if (!f.good() || epoch < 0)
            {
              cerr << "ERROR (reading model): "
                   << "Invalid epoch number: " << epoch << endl;
              exit(10);
            }
        }
      else if (c == 'W')
        {
          w.clear();
          f >> w;
          if (! f.good() || w.size() != dict.nParams())
            {
              cerr << "ERROR (reading model): "
                   << "Invalid weight vector size: " << w.size() << endl;
              exit(10);
            }
          wnorm = dot(w,w);
          wscale = 1.0;
        }
      else if (c == 'L')
        {
          lambda = -1;
          f >> lambda;
          if (! f.good() || lambda<=0)
            {
              cerr << "ERROR (reading model): "
                   << "Invalid lambda: " << lambda << endl;
              exit(10);
            }
        }
      else
        {
          cerr << "ERROR (reading model): "
               << "Unrecognized line: '" << c << "'" << endl;
          exit(10);
        }
    }
  if (! wscale)
    {
      cerr << "ERROR (reading model): "
           << "This model file does not contain weights. " << endl;
      exit(10);
      
    }
}


istream& 
operator>> (istream &f, CrfSgd &d )
{
  d.load(f);
  return f;
}


void
CrfSgd::rescale()
{
  if (wscale != 1.0)
    {
      w.scale(wscale);
      wscale = 1;
    }
}


void
CrfSgd::save(ostream &f) const
{
  // rescale weights according to wscale
  if (wscale != 1.0)
    const_cast<CrfSgd*>(this)->rescale();
  // save stuff
  f << dict;
  f << "L" << lambda << endl;
  f << "T" << t << endl;
  f << "E" << epoch << endl;
  f << "W" << w << endl;
}


ostream& 
operator<<(ostream &f, const CrfSgd &d)
{
  d.save(f);
  return f;
}


void 
CrfSgd::initialize(const char *tfname, const char *dfname, 
                   double c, int cutoff)
{
  t = 0;
  epoch = 0;
  int n = dict.initFromData(tfname, dfname, cutoff);
  lambda = 1 / (c * n);
  if (verbose)
    cout << "Using c=" << c << ", i.e. lambda=" << lambda << endl;
  w.clear();
  w.resize(dict.nParams());
  wscale = 1.0;
  wnorm = 0;
}


double 
CrfSgd::findObjBySampling(const dataset_t &data, const ivec_t &sample)
{
  double loss = 0;
  int n = sample.size();
  for (int i=0; i<n; i++)
    {
      int j = sample[i];
      Scorer scorer(data[j], dict, w, wscale);
      loss += scorer.scoreForward() - scorer.scoreCorrect();
    }
  return loss / n + 0.5 * wnorm * lambda;
}


double
CrfSgd::tryEtaBySampling(const dataset_t &data, const ivec_t &sample,
                         double eta)
{
  FVector savedW = w;
  double savedWScale = wscale;
  double savedWNorm = wnorm;
  int i, n = sample.size();
  for (i=0; i<n; i++)
    {
      int j = sample[i];
      TScorer scorer(data[j], dict, w, wscale, eta);
      scorer.gradCorrect(+1);
      scorer.gradForward(-1);
      wscale *= (1 - eta * lambda);
    }
  wnorm = dot(w,w) * wscale * wscale;
  double obj = findObjBySampling(data, sample);
  w = savedW;
  wscale = savedWScale;
  wnorm = savedWNorm;
  return obj;
}


void 
CrfSgd::adjustEta(double eta)
{  
  t = 1 / (eta * lambda);
  if (verbose)
    cout << " taking eta=" << eta << "  t0=" << t;
}


void 
CrfSgd::adjustEta(const dataset_t &data, int samples, 
                  double seta, Timer *tm)
{
  ivec_t sample;
  if (verbose)
    cout << "[Calibrating] --  " << samples << " samples" << endl;
  assert(samples > 0);
  assert(dict.nOutputs() > 0);
  // choose sample
  int datasize = data.size();
  if (samples < datasize)
    for (int i=0; i<samples; i++)
      sample.push_back((int)((double)rand()*datasize/RAND_MAX));
  else
    for (int i=0; i<datasize; i++)
      sample.push_back(i);
  // initial obj
  double sobj = findObjBySampling(data, sample);
  cout << " initial objective=" << sobj << endl;
  // empirically find eta that works best
  double besteta = 1;
  double bestobj = sobj;
  double eta = seta;
  int totest = 10;
  double factor = 2;
  bool phase2 = false;
  while (totest > 0 || !phase2)
    {
      double obj = tryEtaBySampling(data, sample, eta);
      bool okay = (obj < sobj);
      if (verbose)
        {
          cout << " trying eta=" << eta << "  obj=" << obj;
          if (okay)
            cout << " (possible)" << endl;
          else
            cout << " (too large)" << endl;
        }
      if (okay)
        {
          totest -= 1;
          if (obj < bestobj) {
            bestobj = obj;
            besteta = eta;
          }
        }
      if (! phase2)
        {
          if (okay)
            eta = eta * factor;
          else {
            phase2 = true;
            eta = seta;
          }
        }
      if (phase2)
        eta = eta / factor;
    }
  // take it on the safe side (implicit regularization)
  besteta /= factor;
  // set initial t
  adjustEta(besteta);
  // message
  if  (tm && verbose)
    cout << " time=" << tm->elapsed() << "s.";
  if (verbose)
    cout << endl;
}


void 
CrfSgd::train(const dataset_t &data, int epochs, Timer *tm)
{
  if (t <= 0)
    {
      cerr << "ERROR (train): "
           << "Must call adjustEta() before train()." << endl;
      exit(10);
    }
  ivec_t shuffle;
  for (unsigned int i=0; i<data.size(); i++)
    shuffle.push_back(i);
  for (int j=0; j<epochs; j++)
    {
      epoch += 1;
      // shuffle examples
      random_shuffle(shuffle.begin(), shuffle.end());
      if (verbose)
        cout << "[Epoch " << epoch << "] --";
      if (verbose)
        cout.flush();
      // perform epoch
      for (unsigned int i=0; i<data.size(); i++)
        {
          double eta = 1/(lambda*t);
          // train
          TScorer scorer(data[shuffle[i]], dict, w, wscale, eta);
          scorer.gradCorrect(+1);
          scorer.gradForward(-1);
          // weight decay
          wscale *= (1 - eta * lambda);
          // iteration done
          t += 1;
        }
      // epoch done
      if (wscale < 1e-5)
        rescale();
      wnorm = dot(w,w) * wscale * wscale;
      cout << " wnorm=" << wnorm;
      if (tm && verbose)
        cout << " time=" << tm->elapsed() << "s.";
      if (verbose)
        cout << endl;
    }
  // this never hurts
  rescale();
}


void
CrfSgd::test(const dataset_t &data, const char *conlleval, Timer *tm)
{
   if (dict.nOutputs() <= 0)
    {
      cerr << "ERROR (test): "
           << "Must call load() or initialize() before test()." << endl;
      exit(10);
    }
   opstream f;
   string evalcommand;
   if (conlleval && conlleval[0] && verbose)
     f.open(conlleval);
   if (verbose)
     cout << " sentences=" << data.size();
   double obj = 0;
   int errors = 0;
   int total = 0;
   for (unsigned int i=0; i<data.size(); i++)
     {
       Scorer scorer(data[i], dict, w, wscale);
       obj += scorer.scoreForward() - scorer.scoreCorrect();
       if (conlleval && conlleval[0] && verbose)
         errors += scorer.test(f);
       else if (conlleval)
         errors += scorer.test(cout);
       else
         errors += scorer.test();
       total += data[i].size();
     }
   obj = obj / data.size();
   if (verbose)
     cout << " loss=" << obj;
   obj += 0.5 * wnorm * lambda;
   double misrate = (double)(errors*100)/(total ? total : 1);
   if (verbose)
     cout << " obj=" << obj 
          << " err=" << errors << " (" << misrate << "%)";
   if (tm && verbose)
     cout << " time=" << tm->elapsed() << "s";
   if (verbose)
     cout << endl;
}





// ============================================================
// Main


string modelFile;
string templateFile;
string trainFile;
string testFile;

const char *conlleval = "./conlleval -q";

double c = 1;
double eta = 0;
int cutoff = 3;
int epochs = 50;
int cepochs = 5;
bool tag = false;

dataset_t train;
dataset_t test;


void 
usage()
{
  cerr 
    << "Usage (training): "
    << "crfsgd [options] model template traindata [devdata]" << endl
    << "Usage (tagging):  "
    << "crfsgd -t model testdata" << endl
    << "Options for training:" << endl
    << " -c <num> : capacity control parameter (1.0)" << endl
    << " -f <num> : threshold on the occurences of each feature (3)" << endl
    << " -r <num> : total number of epochs (50)" << endl
    << " -h <num> : epochs between each testing phase (5)" << endl
    << " -e <cmd> : performance evaluation command (conlleval -q)" << endl
    << " -s <num> : initial learning rate" << endl
    << " -q       : silent mode" << endl;
  exit(10);
}

void 
parseCmdLine(int argc, char **argv)
{
  for (int i=1; i<argc; i++)
    {
      const char *s = argv[i];
      if (s[0]=='-')
        {
          while (s[0] == '-')
            s++;
          if (tag || s[1])
            usage();
          if (s[0] == 't')
            {
              if (i == 1)
                tag = true;
              else
                usage();
            }
          else if (s[0] == 'q')
            verbose = false;
          else if (++i >= argc)
            usage();
          else if (s[0 ] == 'c')
            {
              c = atof(argv[i]);
              if (c <= 0)
                {
                  cerr << "ERROR: "
                       << "Illegal C value: " << c << endl;
                  exit(10);
                }
            }
         else if (s[0 ] == 's')
            {
              eta = atof(argv[i]);
              if (eta <= 0)
                {
                  cerr << "ERROR: "
                       << "Illegal initial learning rate: " << s << endl;
                  exit(10);
                }
            }
          else if (s[0] == 'f')
            {
              cutoff = atoi(argv[i]);
              if (cutoff <= 0 || cutoff > 1000)
                {
                  cerr << "ERROR: " 
                       << "Illegal cutoff value: " << cutoff << endl;
                  exit(10);
                }
            }
          else if (s[0] == 'r')
            {
              epochs = atoi(argv[i]);
              if (epochs <= 0)
                {
                   cerr << "ERROR: " 
                        << "Illegal number of epochs: " << epochs << endl;
                   exit(10);
                }
            }
          else if (s[0] == 'h')
            {
              cepochs = atoi(argv[i]);
              if (cepochs <= 0)
                {
                   cerr << "ERROR: " 
                        << "Illegal number of epochs: " << cepochs << endl;
                   exit(10);
                }
            }
          else if (s[0] == 'e')
            {
              conlleval = argv[i];
              if (!conlleval[0])
                conlleval = 0;
            }
          else
            {
                  cerr << "ERROR: " 
                       << "Unrecognized option: " << argv[i-1] << endl;
                  exit(10);
            }
        }
      else if (tag)
        {
          if (modelFile.empty())
            modelFile = argv[i];
          else if (testFile.empty())
            testFile = argv[i];
          else 
            usage();
        }
      else
        {
          if (modelFile.empty())
            modelFile = argv[i];
          else if (templateFile.empty())
            templateFile = argv[i];
          else if (trainFile.empty())
            trainFile = argv[i];
          else if (testFile.empty())
            testFile = argv[i];
          else 
            usage();
        }
    }
  if (tag)
    {
      verbose = false;
      if (modelFile.empty() || 
          testFile.empty())
        usage();
    }
  else
    {
      if (modelFile.empty() || 
          templateFile.empty() ||
          trainFile.empty())
        usage();
    }
}



int 
main(int argc, char **argv)
{
  // parse args
  parseCmdLine(argc, argv);
  // initialize crf
  CrfSgd crf;
  if (tag) 
    {
      igzstream f(modelFile.c_str()); 
      f >> crf;
      loadSentences(testFile.c_str(), crf.getDict(), test);
      // tagging
      crf.test(test, "");
    } 
  else 
    {
      crf.initialize(templateFile.c_str(), trainFile.c_str(), c, cutoff);
      loadSentences(trainFile.c_str(), crf.getDict(), train);
      if (! testFile.empty())
        loadSentences(testFile.c_str(), crf.getDict(), test);
      // training
      Timer tm;
      tm.start();
      if (eta > 0)
        crf.adjustEta(eta);
      else
        crf.adjustEta(train, 1000, 0.1, &tm);
      tm.stop();
      while (crf.getEpoch() < epochs)
        {
          tm.start();
          int ce = cepochs; // (crf.getEpoch() < cepochs) ? 1 : cepochs;
          crf.train(train, ce, &tm);
          tm.stop();
          if (verbose)
            {
              cout << "Training perf:";
              crf.test(train, conlleval);
            }
          if (verbose && test.size())
            {
              cout << "Testing perf:";
              crf.test(test, conlleval);
            }
        }
      if (verbose)
        cout << "Saving model file " << modelFile << "." << endl;
      ogzstream f(modelFile.c_str()); 
      f << crf; 
      if (verbose)
        cout << "Done!  " << tm.elapsed() << " seconds." << endl;
    }
  return 0;
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
