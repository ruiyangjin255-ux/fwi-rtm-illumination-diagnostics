CC = gcc    
CFLAGS = -Wall -O3 -ffast-math -fopenmp -fcommon


src_mod = 2dmod_pml.o    \
		  readpar.o      \
		  wavelet.o      \
		  update_p_pml.o \
		  pml_boundary.o \
		  globalvar.o \
		  pad.o

2dmod_pml.o : 2dmod_pml.c fd.h globalvar.h
	$(CC) $(CFLAGS) -c 2dmod_pml.c 
readpar.o : readpar.c fd.h globalvar.h
	$(CC) $(CFLAGS) -c readpar.c 
wavelet.o : wavelet.c fd.h globalvar.h
	$(CC) $(CFLAGS) -c wavelet.c  
update_p_pml.o: update_p_pml.c fd.h globalvar.h
	$(CC) $(CFLAGS) -c update_p_pml.c
pad.o: pad.c fd.h globalvar.h
	$(CC) $(CFLAGS) -c pad.c  
pml_boundary.o: pml_boundary.c fd.h globalvar.h
	$(CC) $(CFLAGS) -c pml_boundary.c  
globalvar.o: globalvar.c globalvar.h
	$(CC) $(CFLAGS) -c globalvar.c  

mod : $(src_mod)
	$(CC) $(CFLAGS) -o ../bin/mod $(src_mod) -lm

.PHONY : clean
clean :
	-rm ../bin/mod

.PHONY : all 
all :  mod 
