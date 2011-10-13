# General makefile

MAKE=make
SHELL=/bin/sh

SUBDIRS=svm crf


world: check all
	@echo "================================================"
	@echo "CONGRATULATIONS: The compilation was successful."
	@echo "To know what to do next, check the README file."
	@echo "================================================"


all clean:
	@for n in ${SUBDIRS} ; \
	  do ( cd $$n && ${MAKE} ${@}) || exit ; done

all: check

check:
	@if [ -r data/rcv1/rcv1-v2.topics.qrels.gz ] ; then : ; else \
	  echo "=======================================" ; \
	  echo "ATTENTION: Missing data files" ; \
	  echo "You should have read the README file!" ; \
	  echo "=======================================" ; \
	fi


.PHONY: world all depend check