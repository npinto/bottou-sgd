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


#ifndef PSTREAM_H
#define PSTREAM_H 1

#include <iostream>
#include <fstream>
#include <cstdio>

class pstreambuf : public std::streambuf 
{
 private:
  static const int bsize = 512;
  char buffer[bsize];
  std::FILE *f;
  int  mode;
 public:
  pstreambuf() : f(0), mode(0) { 
    setp( buffer, buffer+bsize-1 ); 
    setg( buffer+4, buffer+4, buffer+4 );
  }
  int is_open() { return !!f; }
  pstreambuf* open(const char *cmd, int open_mode);
  pstreambuf* close();
  ~pstreambuf() { close(); }
  virtual int overflow( int c = EOF);
  virtual int underflow();
  virtual int sync();
};


class pstreambase : virtual public std::ios {
 protected:
  pstreambuf buf;
 public:
  pstreambase() { init(&buf); }
  pstreambase(const char *cmd, int open_mode);
  ~pstreambase();
  void open(const char *cmd, int open_mode);
  void close();
  pstreambuf* rdbuf() { return &buf; }
};

// ----------------------------------------------------------------------------
// User classes. Use ipstream and opstream analogously to ifstream and
// ofstream respectively. They read and write files using popen().
// ----------------------------------------------------------------------------

class ipstream : public pstreambase, public std::istream {
 public:
  ipstream() : std::istream( &buf) {} 
  ipstream( const char* cmd, int open_mode = std::ios::in)
    : pstreambase(cmd, open_mode), std::istream( &buf) {}  
  pstreambuf* rdbuf() { return pstreambase::rdbuf(); }
  void open( const char* cmd, int open_mode = std::ios::in) {
    pstreambase::open(cmd, open_mode);
  }
};

class opstream : public pstreambase, public std::ostream {
 public:
  opstream() : std::ostream( &buf) {}
  opstream( const char *cmd, int mode = std::ios::out)
    : pstreambase(cmd, mode), std::ostream( &buf) {}  
  pstreambuf* rdbuf() { return pstreambase::rdbuf(); }
  void open( const char *cmd, int open_mode = std::ios::out) {
    pstreambase::open( cmd, open_mode);
  }
};

#endif

/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" "std::\\sw+")
   End:
   ------------------------------------------------------------- */
