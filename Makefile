CC ?= cc
CFLAGS ?= -O3 -std=c11 -D_XOPEN_SOURCE=700 -Wall -Wextra -Werror -pedantic
LDFLAGS ?= -lm

SRC := $(wildcard src/*.c)
OBJ := $(SRC:.c=.o)

all: qstream

qstream: $(OBJ)
	$(CC) $(CFLAGS) -Iinclude -o $@ $(OBJ) $(LDFLAGS)

src/%.o: src/%.c include/qstream.h
	$(CC) $(CFLAGS) -Iinclude -c $< -o $@

clean:
	rm -f src/*.o qstream demo.qsf

.PHONY: all clean
