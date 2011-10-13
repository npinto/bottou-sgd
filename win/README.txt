
This directory contains a visual c++ project (vc9) for compiling the svm and crf programs.
After compiling, you must copy the executables (found in the Release directory) into
their respective source code directories "svm" and "crf".

The zlib library is a prerequisite:
- Download the source code of zlib from "http://zlib.net".
- Compile using  "nmake -f win32\Makefile.msc OBJA=inffast.obj".
- Copy the files zlib.h zconf.h and zlib.lib into the subdirectory "zlib"

