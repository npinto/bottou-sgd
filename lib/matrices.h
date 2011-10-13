// -*- C++ -*-
// Little library of matrices and sparse matrices
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



// $Id$

#ifndef MATRICES_H
#define MATRICES_H 1

#include <cstring>
#include <iostream>
#include <vector>
#include "wrapper.h"
#include "vectors.h"


class FMatrix
{
 private:
  struct Rep
  {
    int refcount;
    int ncols;
    int nrows;
    std::vector<FVector> rows;
    Rep() : ncols(0), nrows(0) { }
    Rep *copy() { return new Rep(*this); }
  };
  
  Wrapper<Rep> w;
  Rep *rep() { return w.rep(); }
  const Rep *rep() const { return w.rep(); }

 public:
  FMatrix() {}
  FMatrix(int rows, int cols) { resize(rows, cols); }
  int rows() const { return rep()->nrows; }
  int cols() const { return rep()->ncols; }
  void resize(int nrows, int ncols=-1);
  VFloat get(int r, int c) const;
  void set(int r, int c, VFloat v);
  
  FVector& operator[](int r);
  
  const FVector operator[](int r) const {
    const Rep *d = rep();
    if (r<0 || r>=d->nrows)
      return FVector();
    return d->rows[r];
  }
};



class SMatrix
{
 private:
  struct Rep
  {
    int refcount;
    int ncols;
    int nrows;
    std::vector<SVector> rows;
    Rep() : ncols(0), nrows(0) { }
    Rep *copy() { return new Rep(*this); }
  };
  
  Wrapper<Rep> w;
  Rep *rep() { return w.rep(); }
  const Rep *rep() const { return w.rep(); }

 public:
  SMatrix() {}
  SMatrix(int rows, int cols) { resize(rows,cols); }
  int rows() const { return rep()->nrows; }
  int cols() const { return rep()->ncols; }
  void resize(int nrows, int ncols=-1);
  VFloat get(int r, int c) const;
  void set(int r, int c, VFloat v);
  
  SVector& operator[](int r);
  
  const SVector operator[](int r) const {
    const Rep *d = rep();
    if (r<0 || r>=d->nrows)
      return SVector();
    return d->rows[r];
  }
};


#endif

/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" "std::\\sw+")
   End:
   ------------------------------------------------------------- */
