# Define the top directory for MUMPS installation
MUMPS_TOPDIR = /users/vmaillou/src/MUMPS_5.7.3
MUMPS_LIBDIR = $(MUMPS_TOPDIR)/lib
MUMPS_INCLUDEDIR = $(MUMPS_TOPDIR)/include

# Include the MUMPS Makefile.inc
include $(MUMPS_TOPDIR)/Makefile.inc

# Define the compiler and flags
LIBDMUMPS = -L$(MUMPS_LIBDIR) -ldmumps -lmumps_common

default: csr_potrf
.SECONDEXPANSION:

csr_potrf: $$@.o
	$(FL) -o $@ $(OPTL) $@.o $(LIBDMUMPS) $(LORDERINGS) $(LIBS) $(RPATH_OPT) $(LIBBLAS) $(LIBOTHERS)


.SUFFIXES: .c .F .o
.F.o:
	$(FC) $(OPTF) -I. -I$(MUMPS_INCLUDEDIR) -I$(MUMPS_TOPDIR)/src $(INCS) -c $*.F $(OUTF)$*.o
.c.o:
	$(CC) $(OPTC) $(CDEFS) -I. -I$(MUMPS_INCLUDEDIR) -I$(MUMPS_TOPDIR)/src $(INCS) -c $*.c $(OUTC)$*.o

# Clean target
clean:
	rm -f csr_potrf *.o

.PHONY: all clean