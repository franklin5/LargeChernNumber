CC=mpicxx
openmpi = /home/ld7/bin/openmpi-1.8.1/installation/
eigenbin = /home/ld7/bin	
CFLAGS=-c -Wall -I${openmpi}/include -I${eigenbin}
LFLAGS=-limf -lm
all: chern

chern: main.o chern.o 
	$(CC) main.o chern.o -o chern $(LFLAGS)

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

chern.o: chern.cpp
	$(CC) $(CFLAGS) chern.cpp

clean:
	rm -rf *.o chern