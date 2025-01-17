/* mpiPconfig.h.  Generated from mpiPconfig.h.in by configure.  */
/* -*- C -*-

   mpiP MPI Profiler ( http://mpip.sourceforge.net/ )

   Please see COPYRIGHT AND LICENSE information at the end of this file.

   ----- 

   Autoconf configuration macros

   $Id: mpiPconfig.h.in 494 2012-11-28 17:54:33Z chcham $
*/


#ifndef _MPIPCONFIG_H
#define _MPIPCONFIG_H

/* Define if demangle.h is present. */
/* #undef HAVE_DEMANGLE_H */

/* Define if using libdwarf for source lookup */
/* #undef USE_LIBDWARF */

/* Define if using binutils libbfd */
#define ENABLE_BFD 1

/* Define if using libunwind */
/* #undef HAVE_LIBUNWIND */

/* Define if MPI I/O functions are present */
#define HAVE_MPI_IO 1

/* Define if MPI RMA functions are present */
#define HAVE_MPI_RMA 1

/* Define if bfd_boolean not defined */
/* #undef HAVE_BFD_BOOLEAN */

/* Define to activate check for negative time values */
/* #undef MPIP_CHECK_TIME */

/* Define to force use of read_real_time for timing */
/* #undef USE_READ_REAL_TIME */

/* Define to force use of gettimeofday for timing */
/* #undef USE_GETTIMEOFDAY */

/* Define to use clock_gettime for timing */
/* #undef USE_CLOCK_GETTIME */

/* Define to use dclock for timing */
/* #undef USE_DCLOCK */

/* Define to use MPI_Wtime for timing */
#define USE_WTIME 1

/* Define to use rts_get_timebase on BG/L for timing */
/* #undef USE_RTS_GET_TIMEBASE */

/* Define to use _rtc on Cray X1 */
/* #undef USE_RTC */

/* Distinguish available BFD calls between 2.15 and 2.15.96 */
#define HAVE_BFD_GET_SECTION_SIZE 1

/* Address bug in Quadrics MPI Opaque Object translation */
/* #undef HAVE_MPIR_TOPOINTER */

/* Do non-MPI configure for API library only */
/* #undef ENABLE_API_ONLY */

/* Have collective report generation be default */
/* #undef COLLECTIVE_REPORT_DEFAULT */

/* Determine whether to include weak symbol include files */
/* #undef ENABLE_FORTRAN_WEAK_SYMS */

/* Use glibc backtrace for stack trace */
#define USE_BACKTRACE 1

/* Use setjmp to get stack pointer for naive stack unwinding */
/* #undef USE_SETJMP */

/* Add mread_real_time declaration if needed. */
/* #undef NEED_MREAD_REAL_TIME_DECL */

/* Use /proc/self/maps to get SO info for source lookup. */
#define SO_LOOKUP 1

/* Set the default report format (concise or verbose). */
#define DEFAULT_REPORT_FORMAT mpiPi_style_verbose

/* Specify use of "const"s, as per the MPI-3 specification.  */
/* #undef USE_MPI3_CONSTS */

#endif /* _CONFIG_H */

/* 

<license>

Copyright (c) 2006, The Regents of the University of California. 
Produced at the Lawrence Livermore National Laboratory 
Written by Jeffery Vetter and Christopher Chambreau. 
UCRL-CODE-223450. 
All rights reserved. 
 
This file is part of mpiP.  For details, see http://mpip.sourceforge.net/. 
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
 
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the disclaimer below.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the disclaimer (as noted below) in
the documentation and/or other materials provided with the
distribution.

* Neither the name of the UC/LLNL nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OF
THE UNIVERSITY OF CALIFORNIA, THE U.S. DEPARTMENT OF ENERGY OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
 
Additional BSD Notice 
 
1. This notice is required to be provided under our contract with the
U.S. Department of Energy (DOE).  This work was produced at the
University of California, Lawrence Livermore National Laboratory under
Contract No. W-7405-ENG-48 with the DOE.
 
2. Neither the United States Government nor the University of
California nor any of their employees, makes any warranty, express or
implied, or assumes any liability or responsibility for the accuracy,
completeness, or usefulness of any information, apparatus, product, or
process disclosed, or represents that its use would not infringe
privately-owned rights.
 
3.  Also, reference herein to any specific commercial products,
process, or services by trade name, trademark, manufacturer or
otherwise does not necessarily constitute or imply its endorsement,
recommendation, or favoring by the United States Government or the
University of California.  The views and opinions of authors expressed
herein do not necessarily state or reflect those of the United States
Government or the University of California, and shall not be used for
advertising or product endorsement purposes.

</license>

*/
