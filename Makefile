TARGETS = serial_spike_sort gendata
HPPFILES = spike_sort.h cycle_timer.h
OBJS_SPIKE = spike_sort.o
OBJS_GENDATA = gendata.o

CC = g++
LDFLAGS=-lpthread
CFLAGS = -Wall -Wextra -g -O3 -std=c++11

default:	$(TARGETS)
all:	serial_spike_sort
data: gendata

serial_spike_sort: $(OBJS_SPIKE)
	$(CC) $(CFLAGS) -o $@ $(OBJS_SPIKE) $(LDFLAGS)

gendata: $(OBJS_GENDATA)
	$(CC) $(CFLAGS) -o $@ $(OBJS_GENDATA)

%.o: %.cpp $(HPPFILES)
	$(CC) $(CFLAGS) -c -o $@ $<
clean:
	rm -f *.o serial_spike_sort gendata



