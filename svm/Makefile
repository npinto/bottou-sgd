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


L=../lib
CXX=g++
OPT=
OPTS=-g -O2
CXXFLAGS= ${OPTS} ${OPT} -Wall -I$L
LIBS = -lz -lm

PROGRAMS = prep_rcv1 prep_alpha prep_webspam svmsgd svmasgd

OBJS = vectors.o gzstream.o timer.o
INCS = $L/vectors.h $L/gzstream.h $L/timer.h $L/wrapper.h $L/assert.h

all: ${PROGRAMS}

clean:
	-rm ${PROGRAMS} 2>/dev/null
	-rm *.o 2>/dev/null

vectors.o: $L/vectors.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ $L/vectors.cpp

gzstream.o: $L/gzstream.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ $L/gzstream.cpp

timer.o: $L/timer.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ $L/timer.cpp

data.o: data.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ data.cpp

prep_rcv1.o: prep_rcv1.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ prep_rcv1.cpp

prep_alpha.o: prep_alpha.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ prep_alpha.cpp

prep_webspam.o: prep_webspam.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ prep_webspam.cpp

svmsgd.o: svmsgd.cpp data.h loss.h ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ svmsgd.cpp

svmasgd.o: svmasgd.cpp data.h loss.h ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ svmasgd.cpp

prep_rcv1: prep_rcv1.o ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ prep_rcv1.o ${OBJS} ${LIBS}

prep_alpha: prep_alpha.o ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ prep_alpha.o ${OBJS} ${LIBS}

prep_webspam: prep_webspam.o ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ prep_webspam.o ${OBJS} ${LIBS}

svmsgd: svmsgd.o data.o ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ svmsgd.o data.o ${OBJS} ${LIBS}

svmasgd: svmasgd.o data.o ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ svmasgd.o data.o ${OBJS} ${LIBS}



