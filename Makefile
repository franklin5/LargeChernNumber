CC=mpicxx
eigenbin = /home/ld7/bin	
CFLAGS=-c -Wall -I${eigenbin}
LFLAGS=-limf -lm
all: chern

chern: main.o chern.o 
	$(CC) main.o chern.o -o chern $(LFLAGS)

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

chern.o: chern.cpp
	$(CC) $(CFLAGS) chern.cpp

touch: 
	touch *.cpp *.h
clean:
	rm *.o chern *~ *#