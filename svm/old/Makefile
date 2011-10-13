# SVM with stochastic gradient

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


L=../../lib


CXX=g++
OPTS=-g -O2
OPT=
CXXFLAGS= ${OPTS} ${OPT} -Wall -I$L
LIBS = -lz -lm

PROGRAMS = svmsgd2 svmsgdqn svmcg svmolbfgs

all: ${PROGRAMS}

clean:
	-rm ${PROGRAMS} 2>/dev/null
	-rm *.o 2>/dev/null

svmsgd2: svmsgd2.o vectors.o gzstream.o timer.o
	${CXX} ${CXXFLAGS} -o $@ svmsgd2.o vectors.o gzstream.o timer.o ${LIBS}

svmsgdqn: svmsgdqn.o vectors.o gzstream.o timer.o
	${CXX} ${CXXFLAGS} -o $@ svmsgd2.o vectors.o gzstream.o timer.o ${LIBS}

svmolbfgs: svmolbfgs.o vectors.o gzstream.o timer.o
	${CXX} ${CXXFLAGS} -o $@ svmolbfgs.o vectors.o gzstream.o timer.o ${LIBS}

svmcg: svmcg.o vectors.o gzstream.o timer.o
	${CXX} ${CXXFLAGS} -o $@ svmcg.o vectors.o gzstream.o timer.o ${LIBS}

preprocess.o: preprocess.cpp $L/vectors.h  $L/gzstream.h
	${CXX} ${CXXFLAGS} -c -o $@ preprocess.cpp

svmsgd2.o: svmsgd2.cpp $L/vectors.h  $L/gzstream.h $L/timer.h
	${CXX} ${CXXFLAGS} -c -o $@ svmsgd2.cpp

svmsgdqn.o: svmsgdqn.cpp $L/vectors.h  $L/gzstream.h $L/timer.h
	${CXX} ${CXXFLAGS} -c -o $@ svmsgdqn.cpp

svmolbfgs.o: svmolbfgs.cpp $L/vectors.h  $L/gzstream.h $L/timer.h
	${CXX} ${CXXFLAGS} -c -o $@ svmolbfgs.cpp

svmcg.o: svmcg.cpp $L/vectors.h  $L/gzstream.h $L/timer.h
	${CXX} ${CXXFLAGS} -c -o $@ svmcg.cpp

vectors.o: $L/vectors.cpp $L/vectors.h  $L/wrapper.h
	${CXX} ${CXXFLAGS} -c -o $@ $L/vectors.cpp

gzstream.o: $L/gzstream.cpp $L/gzstream.h 
	${CXX} ${CXXFLAGS} -c -o $@ $L/gzstream.cpp

timer.o: $L/timer.cpp $L/timer.h 
	${CXX} ${CXXFLAGS} -c -o $@ $L/timer.cpp




