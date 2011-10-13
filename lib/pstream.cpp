// -*- C++ -*-
// Stream that uses popen/pclose internally
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


#include "pstream.h"
#include <cstdio>
#include <cstring>


pstreambuf* 
pstreambuf::open( const char *cmd, int open_mode)
{
  if (f)
    return 0;
  mode = open_mode;
  if ((mode & std::ios::ate) || (mode & std::ios::app)
      || ((mode & std::ios::in) && (mode & std::ios::out)))
    return 0;
  char fmode[10];
  char *fmodeptr = fmode;
  if ( mode & std::ios::in)
    *fmodeptr++ = 'r';
  else if ( mode & std::ios::out)
    *fmodeptr++ = 'w';
#ifdef WIN32
  if (mode & std::ios::binary)
    *fmodeptr++ = 'b';
  *fmodeptr = '\0';
  f = ::_popen(cmd, fmode);
#else
  *fmodeptr = '\0';
  f = ::popen(cmd, fmode);
#endif
  if (f == 0)
    return 0;
  return this;
}


pstreambuf* 
pstreambuf::close() 
{
  if (f)
    {
      sync();
#ifdef WIN32
      ::_pclose(f);
#else
      ::pclose(f);
#endif
      f = 0;
      return this;
    }
  return 0;
}


int 
pstreambuf::underflow() 
{ // used for input buffer only
  if ( gptr() && ( gptr() < egptr()))
    return *reinterpret_cast<unsigned char *>( gptr());
  if ( ! (mode & std::ios::in) || ! f)
    return EOF;
  int n_putback = gptr() - eback();
  if ( n_putback > 4)
    n_putback = 4;
  memcpy(buffer + (4 - n_putback), gptr()-n_putback, n_putback);
  int num = std::fread(buffer+4, 1, bsize-4, f);
  if (num <= 0)
    return EOF;
  setg( buffer + (4 - n_putback),   // beginning of putback area
        buffer + 4,                 // read position
        buffer + 4 + num);          // end of buffer
  // return next character
  return *reinterpret_cast<unsigned char *>( gptr());    
}


int 
pstreambuf::overflow(int c) 
{ // used for output buffer only
  if (!(mode & std::ios::out) || !f)
    return EOF;
  if (c != EOF) {
    *pptr() = c;
    pbump(1);
  }
  if (! sync())
    return c;
  return EOF;
}


int 
pstreambuf::sync() {
  if ( pptr() && pptr() > pbase()) {
    int w = pptr() - pbase();
    if (std::fwrite( pbase(), 1, w, f ) != (size_t)w)
      return EOF;
    pbump( -w);
  }
  return 0;
}


pstreambase::pstreambase( const char* cmd, int mode) {
    init(&buf);
    open(cmd, mode);
}


pstreambase::~pstreambase() {
    buf.close();
}


void 
pstreambase::open( const char* cmd, int open_mode) {
  if (! buf.open(cmd, open_mode))
    setstate( std::ios::badbit);
}


void 
pstreambase::close() {
  if (buf.is_open())
    if (! buf.close())
      setstate(std::ios::badbit);
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */
