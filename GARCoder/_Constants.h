#pragma once

#define AC_MinLength 0x01000000U			  // threshold for renormalization
#define AC_MaxLength 0xFFFFFFFFU			  // maximum interval length

#define BM_LengthShift 13					  // length bits discarded before multiplication
#define BM_MaxCount  (1 << BM_LengthShift)   // for adaptive models

#define CM_LengthShift 15
#define CM_MaxCount (1 << CM_LengthShift)

#define THREADS_PER_BLOCK 64
