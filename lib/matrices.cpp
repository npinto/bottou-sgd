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


#include "assert.h"
#include <matrices.h>


void 
FMatrix::resize(int nrows, int ncols)
{
  w.detach();
  Rep *d = rep();
  if (nrows >= 0)
    {
      d->rows.resize(nrows);
      d->nrows = nrows;
    }
  if (ncols >= 0)
    {
      d->ncols = ncols;
      for (int i=0; i<d->nrows; i++)
        d->rows[i].resize(ncols);
    }
}


VFloat 
FMatrix::get(int r, int c) const 
{
  const Rep *d = rep();
  if (r>=0 && r<d->nrows)
    return d->rows[r].get(c);
  assert(r >= 0);
  return 0;
}


void 
FMatrix::set(int r, int c, VFloat v) 
{
  w.detach();
  Rep *d = rep();
  if (r>=d->nrows)
    resize(r+1);
  if (c>=d->ncols)
    d->ncols = c+1;
  assert(r >= 0);
  d->rows[r].set(c,v);
}
  

FVector& 
FMatrix::operator[](int r)
{
  w.detach();
  Rep *d = rep();
  if (r>=d->nrows)
    resize(r+1);
  assert(r >= 0);
  return d->rows[r];
}


// ----------------------------------------


void 
SMatrix::resize(int nrows, int ncols)
{
  w.detach();
  Rep *d = rep();
  if (nrows >= 0)
    {
      d->rows.resize(nrows);
      d->nrows = nrows;
    }
  if (ncols >= 0 && ncols < d->ncols)
    {
      d->ncols = ncols;
      for (int i=0; i<d->nrows; i++)
        if (d->rows[i].size() > ncols)
          {
            // truncate
            SVector s = d->rows[i];
            SVector &v = d->rows[i];
            v.clear();
            for (const SVector::Pair *p = s; p->i >= 0 && p->i < ncols; p++)
              v.set(p->i, p->v);
          }
    }
}


VFloat 
SMatrix::get(int r, int c) const 
{
  const Rep *d = rep();
  if (r>=0 && r<d->nrows)
    return d->rows[r].get(c);
  assert(r>=0);
  return 0;
}


void 
SMatrix::set(int r, int c, VFloat v) 
{
  w.detach();
  Rep *d = rep();
  if (r>=d->nrows)
    resize(r+1);
  if (c>=d->ncols)
    d->ncols = c+1;
  assert(r>=0);
  d->rows[r].set(c,v);
}
  

SVector& 
SMatrix::operator[](int r)
{
  w.detach();
  Rep *d = rep();
  if (r>=d->nrows)
    resize(r+1);
  assert(r>=0);
  return d->rows[r];
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */
