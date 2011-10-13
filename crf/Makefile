# CRF with stochastic gradient

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA


L=../lib


CXX=g++
OPTS=-g -O2
CXXFLAGS= ${OPTS} -Wall -I$L
LIBS = -lz -lm

PROGRAMS = crfsgd crfasgd

OBJS = vectors.o matrices.o gzstream.o pstream.o timer.o

all: ${PROGRAMS}

clean:
	-rm ${PROGRAMS} 2>/dev/null
	-rm *.o 2>/dev/null

crfsgd: crfsgd.o ${OBJS}
	-rm $@ 2>/dev/null
	${CXX} ${CXXFLAGS} -o $@ crfsgd.o ${OBJS} ${LIBS}

crfasgd: crfasgd.o ${OBJS}
	-rm $@ 2>/dev/null
	${CXX} ${CXXFLAGS} -o $@ crfasgd.o ${OBJS} ${LIBS}

crfsgd.o: crfsgd.cpp $L/vectors.h  $L/gzstream.h $L/timer.h
	${CXX} ${CXXFLAGS} -c -o $@ crfsgd.cpp

crfasgd.o: crfasgd.cpp $L/vectors.h  $L/gzstream.h $L/timer.h
	${CXX} ${CXXFLAGS} -c -o $@ crfasgd.cpp

vectors.o: $L/vectors.cpp $L/vectors.h  $L/wrapper.h $L/assert.h
	${CXX} ${CXXFLAGS} -c -o $@ $L/vectors.cpp

matrices.o: $L/matrices.cpp $L/matrices.h $L/vectors.h  $L/wrapper.h $L/assert.h
	${CXX} ${CXXFLAGS} -c -o $@ $L/matrices.cpp

gzstream.o: $L/gzstream.cpp $L/gzstream.h $L/assert.h
	${CXX} ${CXXFLAGS} -c -o $@ $L/gzstream.cpp

pstream.o: $L/pstream.cpp $L/pstream.h $L/assert.h
	${CXX} ${CXXFLAGS} -c -o $@ $L/pstream.cpp

timer.o: $L/timer.cpp $L/timer.h $L/assert.h
	${CXX} ${CXXFLAGS} -c -o $@ $L/timer.cpp




