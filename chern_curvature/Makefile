CC=mpicxx
eigenbin = /home/ld7/bin	
CFLAGS=-c -Wall -I${eigenbin}
LFLAGS=-limf -lm
all: chern

chern: main.o chern.o lgwt.o
	$(CC) main.o chern.o lgwt.o -o chern $(LFLAGS)

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

chern.o: chern.cpp
	$(CC) $(CFLAGS) chern.cpp

lgwt.o: lgwt.cpp
	$(CC) $(CFLAGS) lgwt.cpp

touch: 
	touch *.h
clean:
	rm *.o chern *~ *#