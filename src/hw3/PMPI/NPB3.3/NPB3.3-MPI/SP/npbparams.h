c NPROCS = 64 CLASS = A
c  
c  
c  This file is generated automatically by the setparams utility.
c  It sets the number of processors and the class of the NPB
c  in this directory. Do not modify it by hand.
c  
        integer maxcells, problem_size, niter_default
        parameter (maxcells=8, problem_size=64, niter_default=400)
        double precision dt_default
        parameter (dt_default = 0.0015d0)
        logical  convertdouble
        parameter (convertdouble = .false.)
        character*11 compiletime
        parameter (compiletime='28 Feb 2016')
        character*3 npbversion
        parameter (npbversion='3.3')
        character*6 cs1
        parameter (cs1='mpif77')
        character*9 cs2
        parameter (cs2='$(MPIF77)')
        character*6 cs3
        parameter (cs3='(none)')
        character*20 cs4
        parameter (cs4='-I/usr/local/include')
        character*2 cs5
        parameter (cs5='-O')
        character*2 cs6
        parameter (cs6='-O')
        character*6 cs7
        parameter (cs7='randi8')
