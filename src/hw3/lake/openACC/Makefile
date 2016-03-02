APP=lake

CC=pgcc
LINKER=pgcc

ODIR = ./
IDIR = ./

VPATH = ./

# ACC=-mp
ACC=-acc -ta=nvidia
CFLAGS=-I$(IDIR) $(ACC) -fast -Minfo -Minline -Msafeptr PGI_ACC_TIME=1

LIBS=-lm

_DEPS = lake.h lake_util.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = lake.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(APP): $(OBJ)
	$(LINKER) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ $(APP)
