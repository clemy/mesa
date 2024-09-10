// from https://github.com/lieff/minih264

#include "h264enc.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if !defined(MINIH264_ONLY_SIMD) && (defined(_M_X64) || defined(_M_ARM64) || defined(__x86_64__) || defined(__aarch64__))
/* x64 always have SSE2, arm64 always have neon, no need for generic code */
//#define MINIH264_ONLY_SIMD
#endif /* SIMD checks... */

#if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))) || ((defined(__i386__) || defined(__x86_64__)) && defined(__SSE2__))
//#define H264E_ENABLE_SSE2 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <emmintrin.h>
#endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
#define H264E_ENABLE_NEON 1
#include <arm_neon.h>
#else
#ifdef MINIH264_ONLY_SIMD
#error MINIH264_ONLY_SIMD used, but SSE/NEON not enabled
#endif
#endif

#ifndef MINIH264_ONLY_SIMD
#define H264E_ENABLE_PLAIN_C 1
#endif


#if defined(__ARMCC_VERSION) || defined(_WIN32) || defined(__EMSCRIPTEN__)
#define __BYTE_ORDER 0
#define __BIG_ENDIAN 1
#elif defined(__linux__) || defined(__CYGWIN__)
#include <endian.h>
#elif defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define __BYTE_ORDER BYTE_ORDER
#define __BIG_ENDIAN BIG_ENDIAN
#elif defined(__OpenBSD__) || defined(__NetBSD__) || defined(__FreeBSD__) || defined(__DragonFly__)
#include <sys/endian.h>
#else
#error platform not supported
#endif

#if defined(__aarch64__) && defined(__clang__)
// uintptr_t broken with aarch64 clang on ubuntu 18
#define uintptr_t unsigned long
#endif
#if defined(__arm__) && defined(__clang__)
#include <arm_acle.h>
#elif defined(__arm__) && defined(__GNUC__) && !defined(__ARMCC_VERSION)
static inline unsigned int __usad8(unsigned int val1, unsigned int val2)
{
    unsigned int result;
    __asm__ volatile ("usad8 %0, %1, %2\n\t"
        : "=r" (result)
        : "r" (val1), "r" (val2));
    return result;
}

static inline unsigned int __usada8(unsigned int val1, unsigned int val2, unsigned int val3)
{
    unsigned int result;
    __asm__ volatile ("usada8 %0, %1, %2, %3\n\t"
        : "=r" (result)
        : "r" (val1), "r" (val2), "r" (val3));
    return result;
}

static inline unsigned int __sadd16(unsigned int val1, unsigned int val2)
{
    unsigned int result;
    __asm__ volatile ("sadd16 %0, %1, %2\n\t"
        : "=r" (result)
        : "r" (val1), "r" (val2));
    return result;
}

static inline unsigned int __ssub16(unsigned int val1, unsigned int val2)
{
    unsigned int result;
    __asm__ volatile ("ssub16 %0, %1, %2\n\t"
        : "=r" (result)
        : "r" (val1), "r" (val2));
    return result;
}

static inline unsigned int __clz(unsigned int val1)
{
    unsigned int result;
    __asm__ volatile ("clz %0, %1\n\t"
        : "=r" (result)
        : "r" (val1));
    return result;
}
#endif

#if defined(_MSC_VER) && _MSC_VER >= 1400
#   define h264e_restrict __restrict
#elif defined(__arm__)
#   define h264e_restrict __restrict
#else
#   define h264e_restrict
#endif
#if defined(_MSC_VER)
#   define ALIGN(n) __declspec(align(n))
#   define ALIGN2(n)
#else
#   define ALIGN(n)
#   define ALIGN2(n) __attribute__((aligned(n)))
#endif

#if __GNUC__ || __clang__
typedef int int_u __attribute__((__aligned__(1)));
#else
typedef int int_u;
#endif

#ifndef MAX
#   define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef MIN
#   define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef ABS
#   define ABS(x)    ((x) >= 0 ? (x) : -(x))
#endif

#define IS_ALIGNED(p, n) (!((uintptr_t)(p) & (uintptr_t)((n) - 1)))

// bit-stream
#if __BYTE_ORDER == __BIG_ENDIAN
#   define SWAP32(x) (uint32_t)(x)
#else
#ifdef _MSC_VER
#   define SWAP32(x) _byteswap_ulong(x)
#elif defined(__GNUC__) || defined(__clang__)
#   define SWAP32(x) __builtin_bswap32(x)
#else
#   define SWAP32(x) (uint32_t)((((x) >> 24) & 0xFF) | (((x) >> 8) & 0xFF00) | (((x) << 8) & 0xFF0000) | ((x & 0xFF) << 24))
#endif
#endif

#define BS_OPEN(bs) uint32_t cache = bs->cache; int shift = bs->shift; uint32_t *buf = bs->buf;
#define BS_CLOSE(bs) bs->cache = cache; bs->shift = shift; bs->buf = buf;
#define BS_PUT(n, val)      \
if ((shift -= n) < 0)       \
{                           \
    cache |= val >> -shift; \
    *buf++ = SWAP32(cache); \
    shift += 32;            \
    cache = 0;              \
}                           \
cache |= (uint32_t)val << shift;

// Quantizer-dequantizer modes
#define QDQ_MODE_INTRA_4   2       // intra 4x4
#define QDQ_MODE_INTER     8       // inter
#define QDQ_MODE_INTRA_16  (8 + 1) // intra 16x16
#define QDQ_MODE_CHROMA    (4 + 1) // chroma

// put most frequently used bits to lsb, to use these as look-up tables
#define AVAIL_TR    8
#define AVAIL_TL    4
#define AVAIL_L     2
#define AVAIL_T     1

typedef uint8_t     pix_t;
typedef uint32_t    bs_item_t;

/**
*   Output bitstream
*/
typedef struct
{
    int         shift;  // bit position in the cache
    uint32_t    cache;  // bit cache
    bs_item_t* buf;  // current position
    bs_item_t* origin; // initial position
} bs_t;

/**
*   Tuple for motion vector, or height/width representation
*/
typedef union
{
    struct
    {
        int16_t x;      // horizontal or width
        int16_t y;      // vertical or height
    } s;
    int32_t u32;        // packed representation
} point_t;

/**
*   Rectangle
*/
typedef struct
{
    point_t tl;         // top-left corner
    point_t br;         // bottom-right corner
} rectangle_t;

/**
*   Quantized/dequantized representation for 4x4 block
*/
typedef struct
{
    int16_t qv[16];     // quantized coefficient
    int16_t dq[16];     // dequantized
} quant_t;

/**
*   Scratch RAM, used only for current MB encoding
*/
typedef struct H264E_scratch_tag
{
    pix_t mb_pix_inp[256];          // Input MB (cached)
    pix_t mb_pix_store[4 * 256];      // Prediction variants

    // Quantized/dequantized
    int16_t dcy[16];                // Y DC // must be before qy
    quant_t qy[16];                 // Y 16x4x4 blocks

    int16_t dcu[16];                // U DC: 4 used + align
    quant_t qu[4];                  // U 4x4x4 blocks

    int16_t dcv[16];                // V DC: 4 used + align
    quant_t qv[4];                  // V 4x4x4 blocks

    //// Quantized DC:
    int16_t quant_dc[16];           // Y
    int16_t quant_dc_u[4];          // U
    int16_t quant_dc_v[4];          // V

    uint16_t nz_mask;               // Bit flags for non-zero 4x4 blocks
} scratch_t;

/**
*   Deblock filter frame context
*/
typedef struct
{
    // Motion vectors for 4x4 MB internal sub-blocks, top and left border,
    // 5x5 array without top-left cell:
    //     T0 T1 T2 T4
    //  L0 i0 i1 i2 i3
    //  L1 ...
    //  ......
    //
    point_t df_mv[5*5 - 1];         // MV for current macroblock and neighbors
    uint8_t *df_qp;                 // QP for current row of macroblocks
    int8_t *mb_type;                // Macroblock type for current row of macroblocks
    uint32_t nzflag;                // Bit flags for non-zero 4x4 blocks (left neighbors)

    // Huffman and deblock uses different nnz...
    uint8_t *df_nzflag;             // Bit flags for non-zero 4x4 blocks (top neighbors), only 4 bits used
} deblock_filter_t;

/**
*    Deblock filter parameters for current MB
*/
typedef struct
{
    uint32_t strength32[4*2];       // Strength for 4 colums and 4 rows
    uint8_t tc0[16*2];              // TC0 parameter for 4 colums and 4 rows
    uint8_t alpha[2*2];             // alpha for border/internals
    uint8_t beta[2*2];              // beta for border/internals
} deblock_params_t;

/**
*   Persistent RAM
*/
typedef struct H264E_persist_tag
{
    H264E_create_param_t param;     // Copy of create parameters
    H264E_io_yuv_t inp;             // Input picture

    struct
    {
        int pic_init_qp;            // Initial QP
    } sps;

    struct
    {
        int num;                    // Frame number
        int nmbx;                   // Frame width, macroblocks
        int nmby;                   // Frame height, macroblocks
        int nmb;                    // Number of macroblocks in frame
        int w;                      // Frame width, pixels
        int h;                      // Frame height, pixels
        rectangle_t mv_limit;       // Frame MV limits = frame + border extension
        rectangle_t mv_qpel_limit;  // Reduced MV limits for qpel interpolation filter
        int cropping_flag;          // Cropping indicator
    } frame;

    struct
    {
        int type;                   // Current slice type (I/P)
        int start_mb_num;           // # of 1st MB in the current slice
    } slice;

    struct
    {
        int x;                      // MB x position (in MB's)
        int y;                      // MB y position (in MB's)
        int num;                    // MB number
        int skip_run;               // Skip run count

        // according to table 7-13
        // -1 = skip, 0 = P16x16, 1 = P16x8, 2=P8x16, 3 = P8x8, 5 = I4x4, >=6 = I16x16
        int type;                   // MB type

        struct
        {
            int pred_mode_luma;     // Intra 16x16 prediction mode
        } i16;

        int8_t i4x4_mode[16];       // Intra 4x4 prediction modes

        int cost;                   // Best coding cost
        int avail;                  // Neighbor availability flags
        point_t mvd[16];            // Delta-MV for each 4x4 sub-part
        point_t mv[16];             // MV for each 4x4 sub-part

        point_t mv_skip_pred;       // Skip MV predictor
    } mb;

    H264E_io_yuv_t ref;             // Current reference picture
    H264E_io_yuv_t dec;             // Reconstructed current macroblock

    struct
    {
        int qp;                     // Current QP

        int prev_qp;                // Previous MB QP

        // Quantizer data, passed to low-level functions
        // layout:
        // multiplier_quant0, multiplier_dequant0,
        // multiplier_quant2, multiplier_dequant2,
        // multiplier_quant1, multiplier_dequant1,
        // rounding_factor_pos,
        // zero_thr_inter
        // zero_thr_inter2
        // ... and same data for chroma
        //uint16_t qdat[2][(6 + 4)];
#define OFFS_RND_INTER 6
#define OFFS_RND_INTRA 7
#define OFFS_THR_INTER 8
#define OFFS_THR2_INTER 9
#define OFFS_THR_1_OFF 10
#define OFFS_THR_2_OFF 18
#define OFFS_QUANT_VECT 26
#define OFFS_DEQUANT_VECT 34
        //struct
        //{
        //    uint16_t qdq[6];
        //    uint16_t rnd[2]; // inter/intra
        //    uint16_t thr[2]; // thresholds
        //    uint16_t zero_thr[2][8];
        //    uint16_t qfull[8];
        //    uint16_t dqfull[8];
        //} qdat[2];
        uint16_t qdat[2][6 + 2 + 2 + 8 + 8 + 8 + 8];
    } rc;

    deblock_filter_t df;            // Deblock filter

    // Speed/quality trade-off
    struct
    {
        int disable_deblock;        // Disable deblock filter flags
    } speed;

    // predictors contexts
    point_t* mv_pred;               // MV for left&top 4x4 blocks
    uint8_t* nnz;                   // Number of non-zero coeffs per 4x4 block for left&top
    int32_t* i4x4mode;              // Intra 4x4 mode for left&top
    pix_t* top_line;                // left&top neighbor pixels

    // output data
    uint8_t* out;                   // Output data storage (pointer to scratch RAM!)
    unsigned int out_pos;           // Output byte position
    bs_t bs[1];                     // Output bitbuffer

    scratch_t* scratch;             // Pointer to scratch RAM
    H264E_run_param_t run_param;    // Copy of run-time parameters

    // Consecutive IDR's must have different idr_pic_id,
    // unless there are some P between them
    uint8_t next_idr_pic_id;

    pix_t* pbest;                   // Macroblock best predictor
    pix_t* ptest;                   // Macroblock predictor under test

    point_t mv_clusters[2];         // MV clusterization for prediction

} h264e_enc_t;

/************************************************************************/
/*      Constants                                                       */
/************************************************************************/

// Tunable constants can be adjusted by the "training" application
#ifndef ADJUSTABLE
#   define ADJUSTABLE static const
#endif

// Huffman encode tables
#define CODE8(val, len) (uint8_t)((val << 4) + len)
#define CODE(val, len) (uint8_t)((val << 4) + (len - 1))

const uint8_t h264e_g_run_before[57] =
{
    15, 17, 20, 24, 29, 35, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    /**** Table #  0 size  2 ****/
    CODE8(1, 1), CODE8(0, 1),
    /**** Table #  1 size  3 ****/
    CODE8(1, 1), CODE8(1, 2), CODE8(0, 2),
    /**** Table #  2 size  4 ****/
    CODE8(3, 2), CODE8(2, 2), CODE8(1, 2), CODE8(0, 2),
    /**** Table #  3 size  5 ****/
    CODE8(3, 2), CODE8(2, 2), CODE8(1, 2), CODE8(1, 3), CODE8(0, 3),
    /**** Table #  4 size  6 ****/
    CODE8(3, 2), CODE8(2, 2), CODE8(3, 3), CODE8(2, 3), CODE8(1, 3), CODE8(0, 3),
    /**** Table #  5 size  7 ****/
    CODE8(3, 2), CODE8(0, 3), CODE8(1, 3), CODE8(3, 3), CODE8(2, 3), CODE8(5, 3), CODE8(4, 3),
    /**** Table #  6 size 15 ****/
    CODE8(7, 3), CODE8(6, 3), CODE8(5, 3), CODE8(4, 3), CODE8(3, 3), CODE8(2,  3), CODE8(1,  3), CODE8(1, 4),
    CODE8(1, 5), CODE8(1, 6), CODE8(1, 7), CODE8(1, 8), CODE8(1, 9), CODE8(1, 10), CODE8(1, 11),
};

const uint8_t h264e_g_total_zeros_cr_2x2[12] =
{
    3, 7, 10,
    /**** Table #  0 size  4 ****/
    CODE8(1, 1), CODE8(1, 2), CODE8(1, 3), CODE8(0, 3),
    /**** Table #  1 size  3 ****/
    CODE8(1, 1), CODE8(1, 2), CODE8(0, 2),
    /**** Table #  2 size  2 ****/
    CODE8(1, 1), CODE8(0, 1),
};

const uint8_t h264e_g_total_zeros[150] =
{
    15, 31, 46, 60, 73, 85, 96, 106, 115, 123, 130, 136, 141, 145, 148,
    /**** Table #  0 size 16 ****/
    CODE8(1, 1), CODE8(3, 3), CODE8(2, 3), CODE8(3, 4), CODE8(2, 4), CODE8(3, 5), CODE8(2, 5), CODE8(3, 6),
    CODE8(2, 6), CODE8(3, 7), CODE8(2, 7), CODE8(3, 8), CODE8(2, 8), CODE8(3, 9), CODE8(2, 9), CODE8(1, 9),
    /**** Table #  1 size 15 ****/
    CODE8(7, 3), CODE8(6, 3), CODE8(5, 3), CODE8(4, 3), CODE8(3, 3), CODE8(5, 4), CODE8(4, 4), CODE8(3, 4),
    CODE8(2, 4), CODE8(3, 5), CODE8(2, 5), CODE8(3, 6), CODE8(2, 6), CODE8(1, 6), CODE8(0, 6),
    /**** Table #  2 size 14 ****/
    CODE8(5, 4), CODE8(7, 3), CODE8(6, 3), CODE8(5, 3), CODE8(4, 4), CODE8(3, 4), CODE8(4, 3), CODE8(3, 3),
    CODE8(2, 4), CODE8(3, 5), CODE8(2, 5), CODE8(1, 6), CODE8(1, 5), CODE8(0, 6),
    /**** Table #  3 size 13 ****/
    CODE8(3, 5), CODE8(7, 3), CODE8(5, 4), CODE8(4, 4), CODE8(6, 3), CODE8(5, 3), CODE8(4, 3), CODE8(3, 4),
    CODE8(3, 3), CODE8(2, 4), CODE8(2, 5), CODE8(1, 5), CODE8(0, 5),
    /**** Table #  4 size 12 ****/
    CODE8(5, 4), CODE8(4, 4), CODE8(3, 4), CODE8(7, 3), CODE8(6, 3), CODE8(5, 3), CODE8(4, 3), CODE8(3, 3),
    CODE8(2, 4), CODE8(1, 5), CODE8(1, 4), CODE8(0, 5),
    /**** Table #  5 size 11 ****/
    CODE8(1, 6), CODE8(1, 5), CODE8(7, 3), CODE8(6, 3), CODE8(5, 3), CODE8(4, 3), CODE8(3, 3), CODE8(2, 3),
    CODE8(1, 4), CODE8(1, 3), CODE8(0, 6),
    /**** Table #  6 size 10 ****/
    CODE8(1, 6), CODE8(1, 5), CODE8(5, 3), CODE8(4, 3), CODE8(3, 3), CODE8(3, 2), CODE8(2, 3), CODE8(1, 4),
    CODE8(1, 3), CODE8(0, 6),
    /**** Table #  7 size  9 ****/
    CODE8(1, 6), CODE8(1, 4), CODE8(1, 5), CODE8(3, 3), CODE8(3, 2), CODE8(2, 2), CODE8(2, 3), CODE8(1, 3),
    CODE8(0, 6),
    /**** Table #  8 size  8 ****/
    CODE8(1, 6), CODE8(0, 6), CODE8(1, 4), CODE8(3, 2), CODE8(2, 2), CODE8(1, 3), CODE8(1, 2), CODE8(1, 5),
    /**** Table #  9 size  7 ****/
    CODE8(1, 5), CODE8(0, 5), CODE8(1, 3), CODE8(3, 2), CODE8(2, 2), CODE8(1, 2), CODE8(1, 4),
    /**** Table # 10 size  6 ****/
    CODE8(0, 4), CODE8(1, 4), CODE8(1, 3), CODE8(2, 3), CODE8(1, 1), CODE8(3, 3),
    /**** Table # 11 size  5 ****/
    CODE8(0, 4), CODE8(1, 4), CODE8(1, 2), CODE8(1, 1), CODE8(1, 3),
    /**** Table # 12 size  4 ****/
    CODE8(0, 3), CODE8(1, 3), CODE8(1, 1), CODE8(1, 2),
    /**** Table # 13 size  3 ****/
    CODE8(0, 2), CODE8(1, 2), CODE8(1, 1),
    /**** Table # 14 size  2 ****/
    CODE8(0, 1), CODE8(1, 1),
};

const uint8_t h264e_g_coeff_token[277 + 18] =
{
    17 + 18, 17 + 18,
    82 + 18, 82 + 18,
    147 + 18, 147 + 18, 147 + 18, 147 + 18,
    212 + 18, 212 + 18, 212 + 18, 212 + 18, 212 + 18, 212 + 18, 212 + 18, 212 + 18, 212 + 18,
    0 + 18,
    /**** Table #  4 size 17 ****/     // offs: 0
    CODE(1, 2), CODE(1, 1), CODE(1, 3), CODE(5, 6), CODE(7, 6), CODE(6, 6), CODE(2, 7), CODE(0, 7), CODE(4, 6),
    CODE(3, 7), CODE(2, 8), CODE(0, 0), CODE(3, 6), CODE(3, 8), CODE(0, 0), CODE(0, 0), CODE(2, 6),
    /**** Table #  0 size 65 ****/     // offs: 17
    CODE(1,  1), CODE(1,  2), CODE(1,  3), CODE(3,  5), CODE(5,  6), CODE(4,  6), CODE(5,  7), CODE(3,  6),
    CODE(7,  8), CODE(6,  8), CODE(5,  8), CODE(4,  7), CODE(7,  9), CODE(6,  9), CODE(5,  9), CODE(4,  8),
    CODE(7, 10), CODE(6, 10), CODE(5, 10), CODE(4,  9), CODE(7, 11), CODE(6, 11), CODE(5, 11), CODE(4, 10),
    CODE(15, 13), CODE(14, 13), CODE(13, 13), CODE(4, 11), CODE(11, 13), CODE(10, 13), CODE(9, 13), CODE(12, 13),
    CODE(8, 13), CODE(14, 14), CODE(13, 14), CODE(12, 14), CODE(15, 14), CODE(10, 14), CODE(9, 14), CODE(8, 14),
    CODE(11, 14), CODE(14, 15), CODE(13, 15), CODE(12, 15), CODE(15, 15), CODE(10, 15), CODE(9, 15), CODE(8, 15),
    CODE(11, 15), CODE(1, 15), CODE(13, 16), CODE(12, 16), CODE(15, 16), CODE(14, 16), CODE(9, 16), CODE(8, 16),
    CODE(11, 16), CODE(10, 16), CODE(5, 16), CODE(0,  0), CODE(7, 16), CODE(6, 16), CODE(0,  0), CODE(0,  0), CODE(4, 16),
    /**** Table #  1 size 65 ****/     // offs: 82
    CODE(3,  2), CODE(2,  2), CODE(3,  3), CODE(5,  4), CODE(11,  6), CODE(7,  5), CODE(9,  6), CODE(4,  4),
    CODE(7,  6), CODE(10,  6), CODE(5,  6), CODE(6,  5), CODE(7,  7), CODE(6,  6), CODE(5,  7), CODE(8,  6),
    CODE(7,  8), CODE(6,  7), CODE(5,  8), CODE(4,  6), CODE(4,  8), CODE(6,  8), CODE(5,  9), CODE(4,  7),
    CODE(7,  9), CODE(6,  9), CODE(13, 11), CODE(4,  9), CODE(15, 11), CODE(14, 11), CODE(9, 11), CODE(12, 11),
    CODE(11, 11), CODE(10, 11), CODE(13, 12), CODE(8, 11), CODE(15, 12), CODE(14, 12), CODE(9, 12), CODE(12, 12),
    CODE(11, 12), CODE(10, 12), CODE(13, 13), CODE(12, 13), CODE(8, 12), CODE(14, 13), CODE(9, 13), CODE(8, 13),
    CODE(15, 13), CODE(10, 13), CODE(6, 13), CODE(1, 13), CODE(11, 13), CODE(11, 14), CODE(10, 14), CODE(4, 14),
    CODE(7, 13), CODE(8, 14), CODE(5, 14), CODE(0,  0), CODE(9, 14), CODE(6, 14), CODE(0,  0), CODE(0,  0), CODE(7, 14),
    /**** Table #  2 size 65 ****/     // offs: 147
    CODE(15,  4), CODE(14,  4), CODE(13,  4), CODE(12,  4), CODE(15,  6), CODE(15,  5), CODE(14,  5), CODE(11,  4),
    CODE(11,  6), CODE(12,  5), CODE(11,  5), CODE(10,  4), CODE(8,  6), CODE(10,  5), CODE(9,  5), CODE(9,  4),
    CODE(15,  7), CODE(8,  5), CODE(13,  6), CODE(8,  4), CODE(11,  7), CODE(14,  6), CODE(9,  6), CODE(13,  5),
    CODE(9,  7), CODE(10,  6), CODE(13,  7), CODE(12,  6), CODE(8,  7), CODE(14,  7), CODE(10,  7), CODE(12,  7),
    CODE(15,  8), CODE(14,  8), CODE(13,  8), CODE(12,  8), CODE(11,  8), CODE(10,  8), CODE(9,  8), CODE(8,  8),
    CODE(15,  9), CODE(14,  9), CODE(13,  9), CODE(12,  9), CODE(11,  9), CODE(10,  9), CODE(9,  9), CODE(10, 10),
    CODE(8,  9), CODE(7,  9), CODE(11, 10), CODE(6, 10), CODE(13, 10), CODE(12, 10), CODE(7, 10), CODE(2, 10),
    CODE(9, 10), CODE(8, 10), CODE(3, 10), CODE(0,  0), CODE(5, 10), CODE(4, 10), CODE(0,  0), CODE(0,  0), CODE(1, 10),
    /**** Table #  3 size 65 ****/     // offs: 212
     3,  1,  6, 11,  0,  5, 10, 15,  4,  9, 14, 19,  8, 13, 18, 23, 12, 17, 22, 27, 16, 21, 26, 31, 20, 25, 30, 35,
    24, 29, 34, 39, 28, 33, 38, 43, 32, 37, 42, 47, 36, 41, 46, 51, 40, 45, 50, 55, 44, 49, 54, 59, 48, 53, 58, 63,
    52, 57, 62,  0, 56, 61,  0,  0, 60
};

/*
    Block scan order
    0 1 4 5
    2 3 6 7
    8 9 C D
    A B E F
*/
static const uint8_t decode_block_scan[16] = { 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15 };

static const uint8_t qpy2qpc[52] = {  // todo: [0 - 9] not used
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
   26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35,
   35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
};

/**
*   Deblock filter constants:
*   <alpha> <thr[1]> <thr[2]> <thr[3]> <beta>
*/
static const uint8_t g_a_tc0_b[52 - 10][5] = {
    {  0,  0,  0,  0,  0},  // 10
    {  0,  0,  0,  0,  0},  // 11
    {  0,  0,  0,  0,  0},  // 12
    {  0,  0,  0,  0,  0},  // 13
    {  0,  0,  0,  0,  0},  // 14
    {  0,  0,  0,  0,  0},  // 15
    {  4,  0,  0,  0,  2},
    {  4,  0,  0,  1,  2},
    {  5,  0,  0,  1,  2},
    {  6,  0,  0,  1,  3},
    {  7,  0,  0,  1,  3},
    {  8,  0,  1,  1,  3},
    {  9,  0,  1,  1,  3},
    { 10,  1,  1,  1,  4},
    { 12,  1,  1,  1,  4},
    { 13,  1,  1,  1,  4},
    { 15,  1,  1,  1,  6},
    { 17,  1,  1,  2,  6},
    { 20,  1,  1,  2,  7},
    { 22,  1,  1,  2,  7},
    { 25,  1,  1,  2,  8},
    { 28,  1,  2,  3,  8},
    { 32,  1,  2,  3,  9},
    { 36,  2,  2,  3,  9},
    { 40,  2,  2,  4, 10},
    { 45,  2,  3,  4, 10},
    { 50,  2,  3,  4, 11},
    { 56,  3,  3,  5, 11},
    { 63,  3,  4,  6, 12},
    { 71,  3,  4,  6, 12},
    { 80,  4,  5,  7, 13},
    { 90,  4,  5,  8, 13},
    {101,  4,  6,  9, 14},
    {113,  5,  7, 10, 14},
    {127,  6,  8, 11, 15},
    {144,  6,  8, 13, 15},
    {162,  7, 10, 14, 16},
    {182,  8, 11, 16, 16},
    {203,  9, 12, 18, 17},
    {226, 10, 13, 20, 17},
    {255, 11, 15, 23, 18},
    {255, 13, 17, 25, 18},
};

/************************************************************************/
/*  Adjustable encoder parameters. Initial MIN_QP values never used     */
/************************************************************************/

ADJUSTABLE uint16_t g_rnd_inter[] = {
    11665, 11665, 11665, 11665, 11665, 11665, 11665, 11665, 11665, 11665,
    11665, 12868, 14071, 15273, 16476,
    17679, 17740, 17801, 17863, 17924,
    17985, 17445, 16904, 16364, 15823,
    15283, 15198, 15113, 15027, 14942,
    14857, 15667, 16478, 17288, 18099,
    18909, 19213, 19517, 19822, 20126,
    20430, 16344, 12259, 8173, 4088,
    4088, 4088, 4088, 4088, 4088,
    4088, 4088,
};

ADJUSTABLE uint16_t g_thr_inter[] = {
    31878, 31878, 31878, 31878, 31878, 31878, 31878, 31878, 31878, 31878,
    31878, 33578, 35278, 36978, 38678,
    40378, 41471, 42563, 43656, 44748,
    45841, 46432, 47024, 47615, 48207,
    48798, 49354, 49911, 50467, 51024,
    51580, 51580, 51580, 51580, 51580,
    51580, 52222, 52864, 53506, 54148,
    54790, 45955, 37120, 28286, 19451,
    10616, 9326, 8036, 6745, 5455,
    4165, 4165,
};

ADJUSTABLE uint16_t g_thr_inter2[] = {
    45352, 45352, 45352, 45352, 45352, 45352, 45352, 45352, 45352, 45352,
    45352, 41100, 36848, 32597, 28345,
    24093, 25904, 27715, 29525, 31336,
    33147, 33429, 33711, 33994, 34276,
    34558, 32902, 31246, 29590, 27934,
    26278, 26989, 27700, 28412, 29123,
    29834, 29038, 28242, 27445, 26649,
    25853, 23440, 21028, 18615, 16203,
    13790, 11137, 8484, 5832, 3179,
    526, 526,
};

ADJUSTABLE uint16_t g_skip_thr_inter[52] =
{
    45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
    45, 45, 45, 44, 44,
    44, 40, 37, 33, 30,
    26, 32, 38, 45, 51,
    57, 58, 58, 59, 59,
    60, 66, 73, 79, 86,
    92, 95, 98, 100, 103,
    106, 200, 300, 400, 500,
    600, 700, 800, 900, 1000,
    1377, 1377,
};

ADJUSTABLE uint16_t g_lambda_q4[52] =
{
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 13, 11, 10, 8,
    7, 11, 15, 20, 24,
    28, 30, 31, 33, 34,
    36, 48, 60, 71, 83,
    95, 95, 95, 96, 96,
    96, 113, 130, 147, 164,
    181, 401, 620, 840, 1059,
    1279, 1262, 1246, 1229, 1213,
    1196, 1196,
};

ADJUSTABLE uint16_t g_lambda_mv_q4[52] =
{
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 14, 15, 15, 16,
    17, 18, 20, 21, 23,
    24, 28, 32, 37, 41,
    45, 53, 62, 70, 79,
    87, 105, 123, 140, 158,
    176, 195, 214, 234, 253,
    272, 406, 541, 675, 810,
    944, 895, 845, 796, 746,
    697, 697,
};

ADJUSTABLE uint16_t g_skip_thr_i4x4[52] =
{
    0,1,2,3,4,5,6,7,8,9,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
    68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
    100, 100,
};

ADJUSTABLE uint16_t g_lambda_i4_q4[] = {
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 31, 34, 38, 41,
    45, 76, 106, 137, 167,
    198, 220, 243, 265, 288,
    310, 347, 384, 421, 458,
    495, 584, 673, 763, 852,
    941, 1053, 1165, 1276, 1388,
    1500, 1205, 910, 614, 319,
    5000, 1448, 2872, 4296, 5720,
    7144, 7144,
};

ADJUSTABLE uint16_t g_lambda_i16_q4[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 3, 7, 10, 14,
    17, 14, 10, 7, 3,
    50, 20, 39, 59, 78,
    98, 94, 89, 85, 80,
    76, 118, 161, 203, 246,
    288, 349, 410, 470, 531,
    592, 575, 558, 540, 523,
    506, 506,
};

ADJUSTABLE uint16_t g_deadzonei[] = {
    3419, 3419, 3419, 3419, 3419, 3419, 3419, 3419, 3419, 3419,
    30550, 8845, 14271, 19698, 25124,
    30550, 29556, 28562, 27569, 26575,
    25581, 25284, 24988, 24691, 24395,
    24098, 24116, 24134, 24153, 24171,
    24189, 24010, 23832, 23653, 23475,
    23296, 23569, 23842, 24115, 24388,
    24661, 19729, 14797, 9865, 4933,
    24661, 3499, 6997, 10495, 13993,
    17491, 17491,
};

#if H264E_ENABLE_PLAIN_C

static uint8_t byteclip_deblock(int x)
{
    if (x > 255)
    {
        return 255;
    }
    if (x < 0)
    {
        return 0;
    }
    return (uint8_t)x;
}

static int clip_range(int range, int src)
{
    if (src > range)
    {
        src = range;
    }
    if (src < -range)
    {
        src = -range;
    }
    return src;
}

static void deblock_chroma(uint8_t *pix, int stride, int alpha, int beta, int thr, int strength)
{
    int p1, p0, q0, q1;
    int delta;

    if (strength == 0)
    {
        return;
    }

    p1 = pix[-2*stride];
    p0 = pix[-1*stride];
    q0 = pix[ 0*stride];
    q1 = pix[ 1*stride];

    if (ABS(p0 - q0) >= alpha || ABS(p1 - p0) >= beta || ABS(q1 - q0) >= beta)
    {
        return;
    }

    if (strength < 4)
    {
        int tC = thr + 1;
        delta = (((q0 - p0)*4) + (p1 - q1) + 4) >> 3;
        delta = clip_range(tC, delta);
        pix[-1*stride] = byteclip_deblock(p0 + delta);
        pix[ 0*stride] = byteclip_deblock(q0 - delta);
    } else
    {
        pix[-1*stride] = (pix_t)((2*p1 + p0 + q1 + 2) >> 2);
        pix[ 0*stride] = (pix_t)((2*q1 + q0 + p1 + 2) >> 2);
    }
}

static void deblock_luma_v(uint8_t *pix, int stride, int alpha, int beta, const uint8_t *pthr, const uint8_t *pstr)
{
    int p2, p1, p0, q0, q1, q2, thr;
    int ap, aq, delta, cloop, i;
    for (i = 0; i < 4; i++)
    {
        cloop = 4;
        if (pstr[i])
        {
            thr = pthr[i];
            do
            {
                p1 = pix[-2];
                p0 = pix[-1];
                q0 = pix[ 0];
                q1 = pix[ 1];

                //if (ABS(p0 - q0) < alpha && ABS(p1 - p0) < beta && ABS(q1 - q0) < beta)
                if (((ABS(p0 - q0) - alpha) & (ABS(p1 - p0) - beta) & (ABS(q1 - q0) - beta)) < 0)
                {
                    int tC = thr;
                    // avoid conditons
                    int sp, sq, d2;
                    p2 = pix[-3];
                    q2 = pix[ 2];
                    ap = ABS(p2 - p0);
                    aq = ABS(q2 - q0);
                    delta = (((q0 - p0)*4) + (p1 - q1) + 4) >> 3;

                    sp = (ap - beta) >> 31;
                    sq = (aq - beta) >> 31;
                    d2 = (((p2 + ((p0 + q0 + 1) >> 1)) >> 1) - p1) & sp;
                    d2 = clip_range(thr, d2);
                    pix[-2] = (pix_t)(p1 + d2);
                    d2 = (((q2 + ((p0 + q0 + 1) >> 1)) >> 1) - q1) & sq;
                    d2 = clip_range(thr, d2);
                    pix[ 1] = (pix_t)(q1 + d2);
                    tC = thr - sp - sq;
                    delta = clip_range(tC, delta);
                    pix[-1] = byteclip_deblock(p0 + delta);
                    pix[ 0] = byteclip_deblock(q0 - delta);
                }
                pix += stride;
            } while (--cloop);
        } else
        {
                pix += 4*stride;
        }
    }
}

static void deblock_luma_h_s4(uint8_t *pix, int stride, int alpha, int beta)
{
    int p3, p2, p1, p0, q0, q1, q2, q3;
    int ap, aq, cloop = 16;
    do
    {
        int abs_p0_q0, abs_p1_p0, abs_q1_q0;
        p1 = pix[-2*stride];
        p0 = pix[-1*stride];
        q0 = pix[ 0*stride];
        q1 = pix[ 1*stride];
        abs_p0_q0 = ABS(p0 - q0);
        abs_p1_p0 = ABS(p1 - p0);
        abs_q1_q0 = ABS(q1 - q0);
        if (abs_p0_q0 < alpha && abs_p1_p0 < beta && abs_q1_q0 < beta)
        {
            int short_p = (2*p1 + p0 + q1 + 2);
            int short_q = (2*q1 + q0 + p1 + 2);

            if (abs_p0_q0 < ((alpha>>2)+2))
            {
                p2 = pix[-3*stride];
                q2 = pix[ 2*stride];
                ap = ABS(p2 - p0);
                aq = ABS(q2 - q0);
                if (ap < beta)
                {
                    int t = p2 + p1 + p0 + q0 + 2;
                    p3 = pix[-4*stride];
                    short_p += t - p1 + q0; //(p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3);
                    short_p >>= 1;
                    pix[-2*stride] = (pix_t)(t >> 2);
                    pix[-3*stride] = (pix_t)((2*p3 + 2*p2 + t + 2) >> 3); //(2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3);
                }
                if (aq < beta)
                {
                    int t = q2 + q1 + p0 + q0 + 2;
                    q3 = pix[ 3*stride];
                    short_q += (t - q1 + p0);//(q2 + 2*q1 + 2*q0 + 2*p0 + p1 + 4)>>3);
                    short_q >>= 1;
                    pix[ 1*stride] = (pix_t)(t >> 2);
                    pix[ 2*stride] = (pix_t)((2*q3 + 2*q2 + t + 2) >> 3); //((2*q3 + 3*q2 + q1 + q0 + p0 + 4) >> 3);
                }
            }
            pix[-1*stride] = (pix_t)(short_p >> 2);
            pix[ 0*stride] = (pix_t)(short_q >> 2);
        }
        pix += 1;
    } while (--cloop);
}

static void deblock_luma_v_s4(uint8_t *pix, int stride, int alpha, int beta)
{
    int p3, p2, p1, p0, q0, q1, q2, q3;
    int ap, aq, cloop = 16;
    do
    {
        p2 = pix[-3];
        p1 = pix[-2];
        p0 = pix[-1];
        q0 = pix[ 0];
        q1 = pix[ 1];
        q2 = pix[ 2];
        if (ABS(p0 - q0) < alpha && ABS(p1 - p0) < beta && ABS(q1 - q0) < beta)
        {
            ap = ABS(p2 - p0);
            aq = ABS(q2 - q0);

            if (ap < beta && ABS(p0 - q0) < ((alpha >> 2) + 2))
            {
                p3 = pix[-4];
                pix[-1] = (pix_t)((p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3);
                pix[-2] = (pix_t)((p2 + p1 + p0 + q0 + 2) >> 2);
                pix[-3] = (pix_t)((2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3);
            } else
            {
                pix[-1] = (pix_t)((2*p1 + p0 + q1 + 2) >> 2);
            }

            if (aq < beta && ABS(p0 - q0) < ((alpha >> 2) + 2))
            {
                q3 = pix[ 3];
                pix[ 0] = (pix_t)((q2 + 2*q1 + 2*q0 + 2*p0 + p1 + 4) >> 3);
                pix[ 1] = (pix_t)((q2 + q1 + p0 + q0 + 2) >> 2);
                pix[ 2] = (pix_t)((2*q3 + 3*q2 + q1 + q0 + p0 + 4) >> 3);
            } else
            {
                pix[ 0] = (pix_t)((2*q1 + q0 + p1 + 2) >> 2);
            }
        }
        pix += stride;
    } while (--cloop);
}

static void deblock_luma_h(uint8_t *pix, int stride, int alpha, int beta, const uint8_t *pthr, const uint8_t *pstr)
{
    int p2, p1, p0, q0, q1, q2;
    int ap, aq, delta, i;
    for (i = 0; i < 4; i++)
    {
        if (pstr[i])
        {
            int cloop = 4;
            int thr = pthr[i];
            do
            {
                p1 = pix[-2*stride];
                p0 = pix[-1*stride];
                q0 = pix[ 0*stride];
                q1 = pix[ 1*stride];

                //if (ABS(p0-q0) < alpha && ABS(p1-p0) < beta && ABS(q1-q0) < beta)
                if (((ABS(p0-q0) - alpha) & (ABS(p1-p0) - beta) & (ABS(q1-q0) - beta)) < 0)
                {
                    int tC = thr;
                    int sp, sq, d2;
                    p2 = pix[-3*stride];
                    q2 = pix[ 2*stride];
                    ap = ABS(p2 - p0);
                    aq = ABS(q2 - q0);
                    delta = (((q0 - p0)*4) + (p1 - q1) + 4) >> 3;

                    sp = (ap - beta) >> 31;
                    d2 = (((p2 + ((p0 + q0 + 1) >> 1)) >> 1) - p1) & sp;
                    d2 = clip_range(thr, d2);
                    pix[-2*stride] = (pix_t)(p1 + d2);

                    sq = (aq - beta) >> 31;
                    d2 = (((q2 + ((p0 + q0 + 1) >> 1)) >> 1) - q1) & sq;
                    d2 = clip_range(thr, d2);
                    pix[ 1*stride] = (pix_t)(q1 + d2);

                    tC = thr - sp - sq;
                    delta = clip_range(tC, delta);

                    pix[-1*stride] = byteclip_deblock(p0 + delta);
                    pix[ 0*stride] = byteclip_deblock(q0 - delta);
                }
                pix += 1;
            } while (--cloop);
        } else
        {
            pix += 4;
        }
    }
}

static void deblock_chroma_v(uint8_t *pix, int32_t stride, int a, int b, const uint8_t *thr, const uint8_t *str)
{
    int i;
    for (i = 0; i < 8; i++)
    {
        deblock_chroma(pix, 1, a, b, thr[i >> 1], str[i >> 1]);
        pix += stride;
    }
}

static void deblock_chroma_h(uint8_t *pix, int32_t stride, int a, int b, const uint8_t *thr, const uint8_t *str)
{
    int i;
    for (i = 0; i < 8; i++)
    {
        deblock_chroma(pix, stride, a, b, thr[i >> 1], str[i >> 1]);
        pix += 1;
    }
}

static void h264e_deblock_chroma(uint8_t *pix, int32_t stride, const deblock_params_t *par)
{
    const uint8_t *alpha = par->alpha;
    const uint8_t *beta  = par->beta;
    const uint8_t *thr   = par->tc0;
    const uint8_t *strength = (uint8_t *)par->strength32;
    int a,b,x,y;
    a = alpha[0];
    b = beta[0];
    for (x = 0; x < 16; x += 8)
    {
        uint32_t str = *(uint32_t*)&strength[x];
        if (str && a)
        {
            deblock_chroma_v(pix + (x >> 1), stride, a, b, thr + x, strength + x);
        }
        a = alpha[1];
        b = beta[1];
    }
    thr += 16;
    strength += 16;
    a = alpha[2];
    b = beta[2];
    for (y = 0; y < 16; y += 8)
    {
        uint32_t str = *(uint32_t*)&strength[y];
        if (str && a)
        {
            deblock_chroma_h(pix, stride, a, b, thr + y, strength + y);
        }
        pix += 4*stride;
        a = alpha[3];
        b = beta[3];
    }
}

static void h264e_deblock_luma(uint8_t *pix, int32_t stride, const deblock_params_t *par)
{
    const uint8_t *alpha = par->alpha;
    const uint8_t *beta  = par->beta;
    const uint8_t *thr   = par->tc0;
    const uint8_t *strength = (uint8_t *)par->strength32;
    int a = alpha[0];
    int b = beta[0];
    int x, y;
    for (x = 0; x < 16; x += 4)
    {
        uint32_t str = *(uint32_t*)&strength[x];
        if ((uint8_t)str == 4)
        {
            deblock_luma_v_s4(pix + x, stride, a, b);
        } else if (str && a)
        {
            deblock_luma_v(pix + x, stride, a, b, thr + x, strength + x);
        }
        a = alpha[1];
        b = beta[1];
    }
    a = alpha[2];
    b = beta[2];
    thr += 16;
    strength += 16;
    for (y = 0; y < 16; y += 4)
    {
        uint32_t str = *(uint32_t*)&strength[y];
        if ((uint8_t)str == 4)
        {
            deblock_luma_h_s4(pix, stride, a, b);
        } else if (str && a)
        {
            deblock_luma_h(pix, stride, a, b, thr + y, strength + y);
        }
        a = alpha[3];
        b = beta[3];
        pix += 4*stride;
    }
}

#undef IS_NULL
#define IS_NULL(p) ((p) < (pix_t *)32)

static uint32_t intra_predict_dc(const pix_t* left, const pix_t* top, int log_side)
{
    unsigned dc = 0, side = 1u << log_side, round = 0;
    do
    {
        if (!IS_NULL(left))
        {
            int cloop = side;
            round += side >> 1;
            do
            {
                dc += *left++;
                dc += *left++;
                dc += *left++;
                dc += *left++;
            } while (cloop -= 4);
        }
        left = top;
        top = NULL;
    } while (left);
    dc += round;
    if (round == side)
        dc >>= 1;
    dc >>= log_side;
    if (!round) dc = 128;
    return dc * 0x01010101;
}

/*
 * Note: To make the code more readable we refer to the neighboring pixels
 * in variables named as below:
 *
 *    UL U0 U1 U2 U3 U4 U5 U6 U7
 *    L0 xx xx xx xx
 *    L1 xx xx xx xx
 *    L2 xx xx xx xx
 *    L3 xx xx xx xx
 */
#define UL edge[-1]
#define U0 edge[0]
#define T1 edge[1]
#define U2 edge[2]
#define U3 edge[3]
#define U4 edge[4]
#define U5 edge[5]
#define U6 edge[6]
#define U7 edge[7]
#define L0 edge[-2]
#define L1 edge[-3]
#define L2 edge[-4]
#define L3 edge[-5]

static void h264e_intra_predict_16x16(pix_t* predict, const pix_t* left, const pix_t* top, int mode)
{
    int cloop = 16;
    uint32_t* d = (uint32_t*)predict;
    assert(IS_ALIGNED(predict, 4));
    assert(IS_ALIGNED(top, 4));
    if (mode != 1)
    {
        uint32_t t0, t1, t2, t3;
        if (mode < 1)
        {
            t0 = ((uint32_t*)top)[0];
            t1 = ((uint32_t*)top)[1];
            t2 = ((uint32_t*)top)[2];
            t3 = ((uint32_t*)top)[3];
        }
        else //(mode == 2)
        {
            t0 = t1 = t2 = t3 = intra_predict_dc(left, top, 4);
        }
        do
        {
            *d++ = t0;
            *d++ = t1;
            *d++ = t2;
            *d++ = t3;
        } while (--cloop);
    }
    else //if (mode == 1)
    {
        do
        {
            uint32_t val = *left++ * 0x01010101u;
            *d++ = val;
            *d++ = val;
            *d++ = val;
            *d++ = val;
        } while (--cloop);
    }
}

static void h264e_intra_predict_chroma(pix_t* predict, const pix_t* left, const pix_t* top, int mode)
{
    int cloop = 8;
    uint32_t* d = (uint32_t*)predict;
    assert(IS_ALIGNED(predict, 4));
    assert(IS_ALIGNED(top, 4));
    if (mode < 1)
    {
        uint32_t t0, t1, t2, t3;
        t0 = ((uint32_t*)top)[0];
        t1 = ((uint32_t*)top)[1];
        t2 = ((uint32_t*)top)[2];
        t3 = ((uint32_t*)top)[3];
        do
        {
            *d++ = t0;
            *d++ = t1;
            *d++ = t2;
            *d++ = t3;
        } while (--cloop);
    }
    else if (mode == 1)
    {
        do
        {
            uint32_t u = left[0] * 0x01010101u;
            uint32_t v = left[8] * 0x01010101u;
            d[0] = u;
            d[1] = u;
            d[2] = v;
            d[3] = v;
            d += 4;
            left++;
        } while (--cloop);
    }
    else //if (mode == 2)
    {
        int ccloop = 2;
        cloop = 2;
        do
        {
            d[0] = d[1] = d[16] = intra_predict_dc(left, top, 2);
            d[17] = intra_predict_dc(left + 4, top + 4, 2);
            if (!IS_NULL(top))
            {
                d[1] = intra_predict_dc(NULL, top + 4, 2);
            }
            if (!IS_NULL(left))
            {
                d[16] = intra_predict_dc(NULL, left + 4, 2);
            }
            d += 2;
            left += 8;
            top += 8;
        } while (--cloop);

        do
        {
            cloop = 12;
            do
            {
                *d = d[-4];
                d++;
            } while (--cloop);
            d += 4;
        } while (--ccloop);
    }
}

static int pix_sad_4(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
                     uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3)
{
#if defined(__arm__)
    int sad = __usad8(r0, x0);
    sad = __usada8(r1, x1, sad);
    sad = __usada8(r2, x2, sad);
    sad = __usada8(r3, x3, sad);
    return sad;
#else
    int c, sad = 0;
    for (c = 0; c < 4; c++)
    {
        int d = (r0 & 0xff) - (x0 & 0xff); r0 >>= 8; x0 >>= 8;
        sad += ABS(d);
    }
    for (c = 0; c < 4; c++)
    {
        int d = (r1 & 0xff) - (x1 & 0xff); r1 >>= 8; x1 >>= 8;
        sad += ABS(d);
    }
    for (c = 0; c < 4; c++)
    {
        int d = (r2 & 0xff) - (x2 & 0xff); r2 >>= 8; x2 >>= 8;
        sad += ABS(d);
    }
    for (c = 0; c < 4; c++)
    {
        int d = (r3 & 0xff) - (x3 & 0xff); r3 >>= 8; x3 >>= 8;
        sad += ABS(d);
    }
    return sad;
#endif
}

static int h264e_intra_choose_4x4(const pix_t *blockin, pix_t *blockpred, int avail, const pix_t *edge, int mpred, int penalty)
{
    int sad, best_sad, best_m = 2;

    uint32_t r0, r1, r2, r3;
    uint32_t x0, x1, x2, x3, x;

    r0 = ((uint32_t *)blockin)[ 0];
    r1 = ((uint32_t *)blockin)[ 4];
    r2 = ((uint32_t *)blockin)[ 8];
    r3 = ((uint32_t *)blockin)[12];
#undef TEST
#define TEST(mode) sad = pix_sad_4(r0, r1, r2, r3, x0, x1, x2, x3); \
        if (mode != mpred) sad += penalty;    \
        if (sad < best_sad)                   \
        {                                     \
            ((uint32_t *)blockpred)[ 0] = x0; \
            ((uint32_t *)blockpred)[ 4] = x1; \
            ((uint32_t *)blockpred)[ 8] = x2; \
            ((uint32_t *)blockpred)[12] = x3; \
            best_sad = sad;                   \
            best_m = mode;                    \
        }

    // DC
    x0 = x1 = x2 = x3 = intra_predict_dc((avail & AVAIL_L) ? &L3 : 0, (avail & AVAIL_T) ? &U0 : 0, 2);
    best_sad = pix_sad_4(r0, r1, r2, r3, x0, x1, x2, x3);
    if (2 != mpred)
    {
        best_sad += penalty;
    }
    ((uint32_t *)blockpred)[ 0] = x0;
    ((uint32_t *)blockpred)[ 4] = x1;
    ((uint32_t *)blockpred)[ 8] = x2;
    ((uint32_t *)blockpred)[12] = x3;


    if (avail & AVAIL_T)
    {
        uint32_t save = *(uint32_t*)&U4;
        if (!(avail & AVAIL_TR))
        {
            *(uint32_t*)&U4 = U3*0x01010101u;
        }

        x0 = x1 = x2 = x3 = *(uint32_t*)&U0;
        TEST(0)

        x  = ((U6 + 3u*U7      + 2u) >> 2) << 24;
        x |= ((U5 + 2u*U6 + U7 + 2u) >> 2) << 16;
        x |= ((U4 + 2u*U5 + U6 + 2u) >> 2) << 8;
        x |= ((U3 + 2u*U4 + U5 + 2u) >> 2);

        x3 = x;
        x = (x << 8) | ((U2 + 2u*U3 + U4 + 2u) >> 2);
        x2 = x;
        x = (x << 8) | ((T1 + 2u*U2 + U3 + 2u) >> 2);
        x1 = x;
        x = (x << 8) | ((U0 + 2u*T1 + U2 + 2u) >> 2);
        x0 = x;
        TEST(3)

        x3 = x1;
        x1 = x0;

        x  = ((U4 + U5 + 1u) >> 1) << 24;
        x |= ((U3 + U4 + 1u) >> 1) << 16;
        x |= ((U2 + U3 + 1u) >> 1) << 8;
        x |= ((T1 + U2 + 1u) >> 1);
        x2 = x;
        x = (x << 8) | ((U0 + T1 + 1) >> 1);
        x0 = x;
        TEST(7)

        *(uint32_t*)&U4 = save;
    }

    if (avail & AVAIL_L)
    {
        x0 = 0x01010101u * L0;
        x1 = 0x01010101u * L1;
        x2 = 0x01010101u * L2;
        x3 = 0x01010101u * L3;
        TEST(1)

        x = x3;
        x <<= 16;
        x |= ((L2 + 3u*L3 + 2u) >> 2) << 8;
        x |= ((L2 + L3 + 1u) >> 1);
        x2 = x;
        x <<= 16;
        x |= ((L1 + 2u*L2 + L3 + 2u) >> 2) << 8;
        x |= ((L1 + L2 + 1u) >> 1);
        x1 = x;
        x <<= 16;
        x |= ((L0 + 2u*L1 + L2 + 2u) >> 2) << 8;
        x |= ((L0 + L1 + 1u) >> 1);
        x0 = x;
        TEST(8)
    }

    if ((avail & (AVAIL_T | AVAIL_L | AVAIL_TL)) == (AVAIL_T | AVAIL_L | AVAIL_TL))
    {
        uint32_t line0, line3;
        x  = ((U3 + 2u*U2 + T1 + 2u) >> 2) << 24;
        x |= ((U2 + 2u*T1 + U0 + 2u) >> 2) << 16;
        x |= ((T1 + 2u*U0 + UL + 2u) >> 2) << 8;
        x |= ((U0 + 2u*UL + L0 + 2u) >> 2);
        line0 = x;
        x0 = x;
        x = (x << 8) | ((UL + 2u*L0 + L1 + 2u) >> 2);
        x1 = x;
        x = (x << 8) | ((L0 + 2u*L1 + L2 + 2u) >> 2);
        x2 = x;
        x = (x << 8) | ((L1 + 2u*L2 + L3 + 2u) >> 2);
        x3 = x;
        line3 = x;
        TEST(4)

        x = x0 << 8;
        x |= ((UL + L0 + 1u) >> 1);
        x0 = x;
        x <<= 8;
        x |= (line3 >> 16) & 0xff;
        x <<= 8;
        x |= ((L0 + L1 + 1u) >> 1);
        x1 = x;
        x <<= 8;
        x |= (line3 >> 8) & 0xff;
        x <<= 8;
        x |= ((L1 + L2 + 1u) >> 1);
        x2 = x;
        x <<= 8;
        x |= line3 & 0xff;
        x <<= 8;
        x |= ((L2 + L3 + 1u) >> 1);
        x3 = x;
        TEST(6)

        x1 = line0;
        x3 = (x1 << 8) | ((line3 >> 8) & 0xFF);

        x  = ((U2 + U3 + 1u) >> 1) << 24;
        x |= ((T1 + U2 + 1u) >> 1) << 16;
        x |= ((U0 + T1 + 1u) >> 1) << 8;
        x |= ((UL + U0 + 1u) >> 1);
        x0 = x;
        x = (x << 8) | ((line3 >> 16) & 0xFF);
        x2 = x;
        TEST(5)
    }
    return best_m + (best_sad << 4);
}

static uint8_t byteclip(int x)
{
    if (x > 255) x = 255;
    if (x < 0) x = 0;
    return (uint8_t)x;
}

static int hpel_lpf(const uint8_t *p, int s)
{
    return p[0] - 5*p[s] + 20*p[2*s] + 20*p[3*s] - 5*p[4*s] + p[5*s];
}

static void copy_wh(const uint8_t *src, int src_stride, uint8_t *dst, int w, int h)
{
    int x, y;
    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            dst [x] = src [x];
        }
        dst += 16;
        src += src_stride;
    }
}

static void hpel_lpf_diag(const uint8_t *src, int src_stride, uint8_t *h264e_restrict dst, int w, int h)
{
    ALIGN(16) int16_t scratch[21 * 16] ALIGN2(16);  /* 21 rows by 16 pixels per row */

    /*
     * Intermediate values will be 1/2 pel at Horizontal direction
     * Starting at (0.5, -2) at top extending to (0.5, height + 3) at bottom
     * scratch contains a 2D array of size (w)X(h + 5)
     */
    int y, x;
    for (y = 0; y < h + 5; y++)
    {
        for (x = 0; x < w; x++)
        {
            scratch[y * w + x] = (int16_t)hpel_lpf(src + (y - 2) * src_stride + (x - 2), 1);
        }
    }

    /* Vertical interpolate */
    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            int pos = y * w + x;
            int HalfCoeff =
                scratch [pos] -
                5 * scratch [pos + 1 * w] +
                20 * scratch [pos + 2 * w] +
                20 * scratch [pos + 3 * w] -
                5 * scratch [pos + 4 * w] +
                scratch [pos + 5 * w];

            HalfCoeff = byteclip((HalfCoeff + 512) >> 10);

            dst [y * 16 + x] = (uint8_t)HalfCoeff;
        }
    }
}

static void hpel_lpf_hor(const uint8_t *src, int src_stride, uint8_t *h264e_restrict dst, int w, int h)
{
    int x, y;
    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            dst [y * 16 + x] = byteclip((hpel_lpf(src + y * src_stride + (x - 2), 1) + 16) >> 5);
        }
    }
}

static void hpel_lpf_ver(const uint8_t *src, int src_stride, uint8_t *h264e_restrict dst, int w, int h)
{
    int y, x;
    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            dst [y * 16 + x] = byteclip((hpel_lpf(src + (y - 2) * src_stride + x, src_stride) + 16) >> 5);
        }
    }
}

static void average_16x16_unalign(uint8_t *dst, const uint8_t *src1, int src1_stride)
{
    int x, y;
    for (y = 0; y < 16; y++)
    {
        for (x = 0; x < 16; x++)
        {
            dst[y * 16 + x] = (uint8_t)(((uint32_t)dst [y * 16 + x] + src1[y*src1_stride + x] + 1) >> 1);
        }
    }
}

static void h264e_qpel_average_wh_align(const uint8_t *src0, const uint8_t *src1, uint8_t *dst, point_t wh)
{
    int w = wh.s.x;
    int h = wh.s.y;
    int x, y;
    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            dst[y * 16 + x] = (uint8_t)((src0[y * 16 + x] + src1[y * 16 + x] + 1) >> 1);
        }
    }
}

static void h264e_qpel_interpolate_luma(const uint8_t *src, int src_stride, uint8_t *h264e_restrict dst, point_t wh, point_t dxdy)
{
    ALIGN(16) uint8_t scratch[16*16] ALIGN2(16);
    //  src += ((dx + 1) >> 2) + ((dy + 1) >> 2)*src_stride;            // dx == 3 ? next row; dy == 3 ? next line
    //  dxdy              actions: Horizontal, Vertical, Diagonal, Average
    //  0 1 2 3 +1        -   ha    h    ha+
    //  1                 va  hva   hda  hv+a
    //  2                 v   vda   d    v+da
    //  3                 va+ h+va h+da  h+v+a
    //  +stride
    int32_t pos = 1 << (dxdy.s.x + 4*dxdy.s.y);
    int dstused = 0;

    if (pos == 1)
    {
        copy_wh(src, src_stride, dst, wh.s.x, wh.s.y);
        return;
    }
    if (pos & 0xe0ee)// 1110 0000 1110 1110
    {
        hpel_lpf_hor(src + ((pos & 0xe000) ? src_stride : 0), src_stride, dst, wh.s.x, wh.s.y);
        dstused++;
    }
    if (pos & 0xbbb0)// 1011 1011 1011 0000
    {
        hpel_lpf_ver(src + ((pos & 0x8880) ? 1 : 0), src_stride, dstused ? scratch : dst, wh.s.x, wh.s.y);
        dstused++;
    }
    if (pos & 0x4e40)// 0100 1110 0100 0000
    {
        hpel_lpf_diag(src, src_stride, dstused ? scratch : dst, wh.s.x, wh.s.y);
        dstused++;
    }
    if (pos & 0xfafa)// 1111 1010 1111 1010
    {
        assert(wh.s.x == 16 && wh.s.y == 16);
        if (dstused == 2)
        {
            point_t p;

            src = scratch;
            src_stride = 16;
            p.u32 = 16 + (16<<16);

            h264e_qpel_average_wh_align(src, dst, dst, p);
            return;
        } else
        {
            src += ((dxdy.s.x + 1) >> 2) + ((dxdy.s.y + 1) >> 2)*src_stride;
        }
        average_16x16_unalign(dst, src, src_stride);
    }
}

static void h264e_qpel_interpolate_chroma(const uint8_t *src, int src_stride, uint8_t *h264e_restrict dst, point_t wh, point_t dxdy)
{
    /* if fractionl mv is not (0, 0) */
    if (dxdy.u32)
    {
        int a = (8 - dxdy.s.x) * (8 - dxdy.s.y);
        int b = dxdy.s.x * (8 - dxdy.s.y);
        int c = (8 - dxdy.s.x) * dxdy.s.y;
        int d = dxdy.s.x * dxdy.s.y;
        int h = wh.s.y;
        do
        {
            int x;
            for (x = 0; x < wh.s.x; x++)
            {
                dst[x] = (uint8_t)((
                   a * src[             x] + b * src[             x + 1] +
                   c * src[src_stride + x] + d * src[src_stride + x + 1] +
                   32) >> 6);
            }
            dst += 16;
            src += src_stride;
        } while (--h);
    } else
    {
        copy_wh(src, src_stride, dst, wh.s.x, wh.s.y);
    }
}

static int sad_block(const pix_t* a, int a_stride, const pix_t* b, int b_stride, int w, int h)
{
    int r, c, sad = 0;
    for (r = 0; r < h; r++)
    {
        for (c = 0; c < w; c++)
        {
            int d = a[c] - b[c];
            sad += ABS(d);
        }
        a += a_stride;
        b += b_stride;
    }
    return sad;
}

static int h264e_sad_mb_unlaign_8x8(const pix_t* a, int a_stride, const pix_t* b, int sad[4])
{
    sad[0] = sad_block(a, a_stride, b, 16, 8, 8);
    sad[1] = sad_block(a + 8, a_stride, b + 8, 16, 8, 8);
    a += 8 * a_stride;
    b += 8 * 16;
    sad[2] = sad_block(a, a_stride, b, 16, 8, 8);
    sad[3] = sad_block(a + 8, a_stride, b + 8, 16, 8, 8);
    return sad[0] + sad[1] + sad[2] + sad[3];
}

static int h264e_sad_mb_unlaign_wh(const pix_t *a, int a_stride, const pix_t *b, point_t wh)
{
    return sad_block(a, a_stride, b, 16, wh.s.x, wh.s.y);
}

static void h264e_copy_8x8(pix_t* d, int d_stride, const pix_t* s)
{
    int cloop = 8;
    assert(IS_ALIGNED(d, 8));
    assert(IS_ALIGNED(s, 8));
    do
    {
        int a = ((const int*)s)[0];
        int b = ((const int*)s)[1];
        ((int*)d)[0] = a;
        ((int*)d)[1] = b;
        s += 16;
        d += d_stride;
    } while (--cloop);
}

static void h264e_copy_16x16(pix_t* d, int d_stride, const pix_t* s, int s_stride)
{
    int cloop = 16;
    assert(IS_ALIGNED(d, 8));
    assert(IS_ALIGNED(s, 8));
    do
    {
        int a = ((const int*)s)[0];
        int b = ((const int*)s)[1];
        int x = ((const int*)s)[2];
        int y = ((const int*)s)[3];
        ((int*)d)[0] = a;
        ((int*)d)[1] = b;
        ((int*)d)[2] = x;
        ((int*)d)[3] = y;
        s += s_stride;
        d += d_stride;
    } while (--cloop);
}
#endif /* H264E_ENABLE_PLAIN_C */

#if H264E_ENABLE_PLAIN_C || (H264E_ENABLE_NEON && !defined(MINIH264_ASM))
static void h264e_copy_borders(unsigned char *pic, int w, int h, int stride, int guard)
{
    int r, rowbytes = w + 2*guard;
    unsigned char *d = pic - guard;
    for (r = 0; r < h; r++, d += stride)
    {
        memset(d, d[guard], guard);
        memset(d + rowbytes - guard, d[rowbytes - guard - 1], guard);
    }
    d = pic - guard - guard*stride;
    for (r = 0; r < guard; r++)
    {
        memcpy(d, pic - guard, rowbytes);
        memcpy(d + (guard + h)*stride, pic - guard + (h - 1)*stride, rowbytes);
        d += stride;
    }
}
#endif /* H264E_ENABLE_PLAIN_C || (H264E_ENABLE_NEON && !defined(MINIH264_ASM)) */

#if H264E_ENABLE_PLAIN_C
#undef TRANSPOSE_BLOCK
#define TRANSPOSE_BLOCK     1
#define UNZIGSAG_IN_QUANT   0
#define SUM_DIF(a, b) { int t = a + b; b = a - b; a = t; }

static int clip_byte(int x)
{
    if (x > 255)
    {
        x = 255;
    }
    else if (x < 0)
    {
        x = 0;
    }
    return x;
}

static void hadamar4_2d(int16_t* x)
{
    int s = 1;
    int sback = 1;
    int16_t tmp[16];
    int16_t* out = tmp;
    int16_t* p = x;
    do
    {
        int cloop = 4;
        do
        {
            int a, b, c, d;
            a = *p; p += 4;//s;
            b = *p; p += 4;//s;
            c = *p; p += 4;//s;
            d = *p; p -= 11;//sback;
            SUM_DIF(a, c);
            SUM_DIF(b, d);
            SUM_DIF(a, b);
            SUM_DIF(c, d);

            *out = (int16_t)a; out += s;
            *out = (int16_t)c; out += s;
            *out = (int16_t)d; out += s;
            *out = (int16_t)b; out += sback;
        } while (--cloop);
        s = 5 - s;
        sback = -11;
        out = x;
        p = tmp;
    } while (s != 1);
}

static void dequant_dc(quant_t* q, int16_t* qval, int dequant, int n)
{
    do q++->dq[0] = (int16_t)(*qval++ * (int16_t)dequant); while (--n);
}

static void quant_dc(int16_t* qval, int16_t* deq, int16_t quant, int n, int round_q18)
{
#if UNZIGSAG_IN_QUANT
    int r_minus = (1 << 18) - round_q18;
    static const uint8_t iscan16[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15 };
    static const uint8_t iscan4[4] = { 0, 1, 2, 3 };
    const uint8_t* scan = n == 4 ? iscan4 : iscan16;
    do
    {
        int v = *qval;
        int r = v < 0 ? r_minus : round_q18;
        deq[*scan++] = *qval++ = (v * quant + r) >> 18;
    } while (--n);
#else
    int r_minus = (1 << 18) - round_q18;
    do
    {
        int v = *qval;
        int r = v < 0 ? r_minus : round_q18;
        *deq++ = *qval++ = (v * quant + r) >> 18;
    } while (--n);
#endif
}

static void hadamar2_2d(int16_t* x)
{
    int a = x[0];
    int b = x[1];
    int c = x[2];
    int d = x[3];
    x[0] = (int16_t)(a + b + c + d);
    x[1] = (int16_t)(a - b + c - d);
    x[2] = (int16_t)(a + b - c - d);
    x[3] = (int16_t)(a - b - c + d);
}

static void h264e_quant_luma_dc(quant_t* q, int16_t* deq, const uint16_t* qdat)
{
    int16_t* tmp = ((int16_t*)q) - 16;
    hadamar4_2d(tmp);
    quant_dc(tmp, deq, qdat[0], 16, 0x20000);//0x15555);
    hadamar4_2d(tmp);
    assert(!(qdat[1] & 3));
    // dirty trick here: shift w/o rounding, since it have no effect  for qp >= 10 (or, to be precise, for qp => 9)
    dequant_dc(q, tmp, qdat[1] >> 2, 16);
}

static int h264e_quant_chroma_dc(quant_t* q, int16_t* deq, const uint16_t* qdat)
{
    int16_t* tmp = ((int16_t*)q) - 16;
    hadamar2_2d(tmp);
    quant_dc(tmp, deq, (int16_t)(qdat[0] << 1), 4, 0xAAAA);
    hadamar2_2d(tmp);
    assert(!(qdat[1] & 1));
    dequant_dc(q, tmp, qdat[1] >> 1, 4);
    return !!(tmp[0] | tmp[1] | tmp[2] | tmp[3]);
}

static const uint8_t g_idx2quant[16] =
{
    0, 2, 0, 2,
    2, 4, 2, 4,
    0, 2, 0, 2,
    2, 4, 2, 4
};

#define TRANSFORM(x0, x1, x2, x3, p, s) { \
    int t0 = x0 + x3;                     \
    int t1 = x0 - x3;                     \
    int t2 = x1 + x2;                     \
    int t3 = x1 - x2;                     \
    (p)[  0] = (int16_t)(t0 + t2);        \
    (p)[  s] = (int16_t)(t1*2 + t3);      \
    (p)[2*s] = (int16_t)(t0 - t2);        \
    (p)[3*s] = (int16_t)(t1 - t3*2);      \
}

static void FwdTransformResidual4x42(const uint8_t* inp, const uint8_t* pred,
    uint32_t inp_stride, int16_t* out)
{
    int i;
    int16_t tmp[16];

#if TRANSPOSE_BLOCK
    // Transform columns
    for (i = 0; i < 4; i++, pred++, inp++)
    {
        int f0 = inp[0] - pred[0];
        int f1 = inp[1 * inp_stride] - pred[1 * 16];
        int f2 = inp[2 * inp_stride] - pred[2 * 16];
        int f3 = inp[3 * inp_stride] - pred[3 * 16];
        TRANSFORM(f0, f1, f2, f3, tmp + i * 4, 1);
    }
    // Transform rows
    for (i = 0; i < 4; i++)
    {
        int d0 = tmp[i + 0];
        int d1 = tmp[i + 4];
        int d2 = tmp[i + 8];
        int d3 = tmp[i + 12];
        TRANSFORM(d0, d1, d2, d3, out + i, 4);
    }

#else
    /* Transform rows */
    for (i = 0; i < 16; i += 4)
    {
        int d0 = inp[0] - pred[0];
        int d1 = inp[1] - pred[1];
        int d2 = inp[2] - pred[2];
        int d3 = inp[3] - pred[3];
        TRANSFORM(d0, d1, d2, d3, tmp + i, 1);
        pred += 16;
        inp += inp_stride;
    }

    /* Transform columns */
    for (i = 0; i < 4; i++)
    {
        int f0 = tmp[i + 0];
        int f1 = tmp[i + 4];
        int f2 = tmp[i + 8];
        int f3 = tmp[i + 12];
        TRANSFORM(f0, f1, f2, f3, out + i, 4);
    }
#endif
}

static void TransformResidual4x4(int16_t* pSrc)
{
    int i;
    int16_t tmp[16];

    /* Transform rows */
    for (i = 0; i < 16; i += 4)
    {
#if TRANSPOSE_BLOCK
        int d0 = pSrc[(i >> 2) + 0];
        int d1 = pSrc[(i >> 2) + 4];
        int d2 = pSrc[(i >> 2) + 8];
        int d3 = pSrc[(i >> 2) + 12];
#else
        int d0 = pSrc[i + 0];
        int d1 = pSrc[i + 1];
        int d2 = pSrc[i + 2];
        int d3 = pSrc[i + 3];
#endif
        int e0 = d0 + d2;
        int e1 = d0 - d2;
        int e2 = (d1 >> 1) - d3;
        int e3 = d1 + (d3 >> 1);
        int f0 = e0 + e3;
        int f1 = e1 + e2;
        int f2 = e1 - e2;
        int f3 = e0 - e3;
        tmp[i + 0] = (int16_t)f0;
        tmp[i + 1] = (int16_t)f1;
        tmp[i + 2] = (int16_t)f2;
        tmp[i + 3] = (int16_t)f3;
    }

    /* Transform columns */
    for (i = 0; i < 4; i++)
    {
        int f0 = tmp[i + 0];
        int f1 = tmp[i + 4];
        int f2 = tmp[i + 8];
        int f3 = tmp[i + 12];
        int g0 = f0 + f2;
        int g1 = f0 - f2;
        int g2 = (f1 >> 1) - f3;
        int g3 = f1 + (f3 >> 1);
        int h0 = g0 + g3;
        int h1 = g1 + g2;
        int h2 = g1 - g2;
        int h3 = g0 - g3;
        pSrc[i + 0] = (int16_t)((h0 + 32) >> 6);
        pSrc[i + 4] = (int16_t)((h1 + 32) >> 6);
        pSrc[i + 8] = (int16_t)((h2 + 32) >> 6);
        pSrc[i + 12] = (int16_t)((h3 + 32) >> 6);
    }
}

static int is_zero(const int16_t* dat, int i0, const uint16_t* thr)
{
    int i;
    for (i = i0; i < 16; i++)
    {
        if ((unsigned)(dat[i] + thr[i & 7]) > (unsigned)2 * thr[i & 7])
        {
            return 0;
        }
    }
    return 1;
}

static int is_zero4(const quant_t* q, int i0, const uint16_t* thr)
{
    return is_zero(q[0].dq, i0, thr) &&
        is_zero(q[1].dq, i0, thr) &&
        is_zero(q[4].dq, i0, thr) &&
        is_zero(q[5].dq, i0, thr);
}

static int zero_smallq(quant_t* q, int mode, const uint16_t* qdat)
{
    int zmask = 0;
    int i, i0 = mode & 1, n = mode >> 1;
    if (mode == QDQ_MODE_INTER || mode == QDQ_MODE_CHROMA)
    {
        for (i = 0; i < n * n; i++)
        {
            if (is_zero(q[i].dq, i0, qdat + OFFS_THR_1_OFF))
            {
                zmask |= (1 << i); //9.19
            }
        }
        if (mode == QDQ_MODE_INTER)   //8.27
        {
            if ((~zmask & 0x0033) && is_zero4(q + 0, i0, qdat + OFFS_THR_2_OFF)) zmask |= 0x33;
            if ((~zmask & 0x00CC) && is_zero4(q + 2, i0, qdat + OFFS_THR_2_OFF)) zmask |= (0x33 << 2);
            if ((~zmask & 0x3300) && is_zero4(q + 8, i0, qdat + OFFS_THR_2_OFF)) zmask |= (0x33 << 8);
            if ((~zmask & 0xCC00) && is_zero4(q + 10, i0, qdat + OFFS_THR_2_OFF)) zmask |= (0x33 << 10);
        }
    }
    return zmask;
}

static int quantize(quant_t* q, int mode, const uint16_t* qdat, int zmask)
{
#if UNZIGSAG_IN_QUANT
#if TRANSPOSE_BLOCK
    // ; Zig-zag scan      Transposed zig-zag
    // ;    0 1 5 6        0 2 3 9
    // ;    2 4 7 C        1 4 8 A
    // ;    3 8 B D        5 7 B E
    // ;    9 A E F        6 C D F
    static const unsigned char iscan16[16] = { 0, 2, 3, 9, 1, 4, 8, 10, 5, 7, 11, 14, 6, 12, 13, 15 };
#else
    static const unsigned char iscan16[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15 };
#endif
#endif
    int i, i0 = mode & 1, ccol, crow;
    int nz_block_mask = 0;
    ccol = mode >> 1;
    crow = ccol;
    do
    {
        do
        {
            int nz_mask = 0;

            if (zmask & 1)
            {
                int32_t* p = (int32_t*)q->qv;
                *p++ = 0; *p++ = 0; *p++ = 0; *p++ = 0;
                *p++ = 0; *p++ = 0; *p++ = 0; *p++ = 0;
            }
            else
            {
                for (i = i0; i < 16; i++)
                {
                    int off = g_idx2quant[i];
                    int v, round = qdat[OFFS_RND_INTER];

                    if (q->dq[i] < 0) round = 0xFFFF - round;

                    v = (q->dq[i] * qdat[off] + round) >> 16;
#if UNZIGSAG_IN_QUANT
                    if (v)
                        nz_mask |= 1 << iscan16[i];
                    q->qv[iscan16[i]] = (int16_t)v;
#else
                    if (v)
                        nz_mask |= 1 << i;
                    q->qv[i] = (int16_t)v;
#endif
                    q->dq[i] = (int16_t)(v * qdat[off + 1]);
                }
            }

            zmask >>= 1;
            nz_block_mask <<= 1;
            if (nz_mask)
                nz_block_mask |= 1;
            q++;
        } while (--ccol);
        ccol = mode >> 1;
    } while (--crow);
    return nz_block_mask;
}

static void transform(const pix_t* inp, const pix_t* pred, int inp_stride, int mode, quant_t* q)
{
    int crow = mode >> 1;
    int ccol = crow;

    do
    {
        do
        {
            FwdTransformResidual4x42(inp, pred, inp_stride, q->dq);
            q++;
            inp += 4;
            pred += 4;
        } while (--ccol);
        ccol = mode >> 1;
        inp += 4 * (inp_stride - ccol);
        pred += 4 * (16 - ccol);
    } while (--crow);
}

static int h264e_transform_sub_quant_dequant(const pix_t* inp, const pix_t* pred, int inp_stride, int mode, quant_t* q, const uint16_t* qdat)
{
    int zmask;
    transform(inp, pred, inp_stride, mode, q);
    if (mode & 1) // QDQ_MODE_INTRA_16 || QDQ_MODE_CHROMA
    {
        int cloop = (mode >> 1) * (mode >> 1);
        short* dc = ((short*)q) - 16;
        quant_t* pq = q;
        do
        {
            *dc++ = pq->dq[0];
            pq++;
        } while (--cloop);
    }
    zmask = zero_smallq(q, mode, qdat);
    return quantize(q, mode, qdat, zmask);
}

static void h264e_transform_add(pix_t* out, int out_stride, const pix_t* pred, quant_t* q, int side, int32_t mask)
{
    int crow = side;
    int ccol = crow;

    assert(IS_ALIGNED(out, 4));
    assert(IS_ALIGNED(pred, 4));
    assert(!(out_stride % 4));

    do
    {
        do
        {
            if (mask >= 0)
            {
                // copy 4x4
                pix_t* dst = out;
                *(uint32_t*)dst = *(uint32_t*)(pred + 0 * 16); dst += out_stride;
                *(uint32_t*)dst = *(uint32_t*)(pred + 1 * 16); dst += out_stride;
                *(uint32_t*)dst = *(uint32_t*)(pred + 2 * 16); dst += out_stride;
                *(uint32_t*)dst = *(uint32_t*)(pred + 3 * 16);
            }
            else
            {
                int i, j;
                TransformResidual4x4(q->dq);
                for (j = 0; j < 4; j++)
                {
                    for (i = 0; i < 4; i++)
                    {
                        int Value = q->dq[j * 4 + i] + pred[j * 16 + i];
                        out[j * out_stride + i] = (pix_t)clip_byte(Value);
                    }
                }
            }
            mask = (uint32_t)mask << 1;
            q++;
            out += 4;
            pred += 4;
        } while (--ccol);
        ccol = side;
        out += 4 * (out_stride - ccol);
        pred += 4 * (16 - ccol);
    } while (--crow);
}
#endif /* H264E_ENABLE_PLAIN_C */

#if H264E_ENABLE_PLAIN_C || (H264E_ENABLE_NEON && !defined(MINIH264_ASM))

#define BS_BITS 32

static void h264e_bs_put_bits(bs_t* bs, unsigned n, unsigned val)
{
    assert(!(val >> n));
    bs->shift -= n;
    assert((unsigned)n <= 32);
    if (bs->shift < 0)
    {
        assert(-bs->shift < 32);
        bs->cache |= val >> -bs->shift;
        *bs->buf++ = SWAP32(bs->cache);
        bs->shift = 32 + bs->shift;
        bs->cache = 0;
    }
    bs->cache |= val << bs->shift;
}

static void h264e_bs_flush(bs_t* bs)
{
    *bs->buf = SWAP32(bs->cache);
}

static unsigned h264e_bs_get_pos_bits(const bs_t* bs)
{
    unsigned pos_bits = (unsigned)((bs->buf - bs->origin) * BS_BITS);
    pos_bits += BS_BITS - bs->shift;
    assert((int)pos_bits >= 0);
    return pos_bits;
}

static unsigned h264e_bs_byte_align(bs_t* bs)
{
    int pos = h264e_bs_get_pos_bits(bs);
    h264e_bs_put_bits(bs, -pos & 7, 0);
    return pos + (-pos & 7);
}

/**
*   Golomb code
*   0 => 1
*   1 => 01 0
*   2 => 01 1
*   3 => 001 00
*   4 => 001 01
*
*   [0]     => 1
*   [1..2]  => 01x
*   [3..6]  => 001xx
*   [7..14] => 0001xxx
*
*/
static void h264e_bs_put_golomb(bs_t* bs, unsigned val)
{
#ifdef __arm__
    int size = 32 - __clz(val + 1);
#else
    int size = 0;
    unsigned t = val + 1;
    do
    {
        size++;
    } while (t >>= 1);
#endif
    h264e_bs_put_bits(bs, 2 * size - 1, val + 1);
}

/**
*   signed Golomb code.
*   mapping to unsigned code:
*       0 => 0
*       1 => 1
*      -1 => 2
*       2 => 3
*      -2 => 4
*       3 => 5
*      -3 => 6
*/
static void h264e_bs_put_sgolomb(bs_t* bs, int val)
{
    val = 2 * val - 1;
    val ^= val >> 31;
    h264e_bs_put_golomb(bs, val);
}

static void h264e_bs_init_bits(bs_t* bs, void* data)
{
    bs->origin = data;
    bs->buf = bs->origin;
    bs->shift = BS_BITS;
    bs->cache = 0;
}

static void h264e_vlc_encode(bs_t* bs, int16_t* quant, int maxNumCoeff, uint8_t* nz_ctx)
{
    int nnz_context, nlevels, nnz; // nnz = nlevels + trailing_ones
    int trailing_ones = 0;
    int trailing_ones_sign = 0;
    uint8_t runs[16];
    uint8_t* prun = runs;
    int16_t* levels;
    int cloop = maxNumCoeff;
    BS_OPEN(bs)

#if H264E_ENABLE_SSE2 || (H264E_ENABLE_PLAIN_C && !H264E_ENABLE_NEON)
        // this branch used with SSE + C configuration
        int16_t zzquant[16];
    levels = zzquant + ((maxNumCoeff == 4) ? 4 : 16);
    if (maxNumCoeff != 4)
    {
        int v;
        if (maxNumCoeff == 16)
        {
            v = quant[15] * 2; if (v) *--levels = (int16_t)v, *prun++ = 16;
            v = quant[11] * 2; if (v) *--levels = (int16_t)v, *prun++ = 15;
            v = quant[14] * 2; if (v) *--levels = (int16_t)v, *prun++ = 14;
            v = quant[13] * 2; if (v) *--levels = (int16_t)v, *prun++ = 13;
            v = quant[10] * 2; if (v) *--levels = (int16_t)v, *prun++ = 12;
            v = quant[7] * 2; if (v) *--levels = (int16_t)v, *prun++ = 11;
            v = quant[3] * 2; if (v) *--levels = (int16_t)v, *prun++ = 10;
            v = quant[6] * 2; if (v) *--levels = (int16_t)v, *prun++ = 9;
            v = quant[9] * 2; if (v) *--levels = (int16_t)v, *prun++ = 8;
            v = quant[12] * 2; if (v) *--levels = (int16_t)v, *prun++ = 7;
            v = quant[8] * 2; if (v) *--levels = (int16_t)v, *prun++ = 6;
            v = quant[5] * 2; if (v) *--levels = (int16_t)v, *prun++ = 5;
            v = quant[2] * 2; if (v) *--levels = (int16_t)v, *prun++ = 4;
            v = quant[1] * 2; if (v) *--levels = (int16_t)v, *prun++ = 3;
            v = quant[4] * 2; if (v) *--levels = (int16_t)v, *prun++ = 2;
            v = quant[0] * 2; if (v) *--levels = (int16_t)v, *prun++ = 1;
        }
        else
        {
            v = quant[15] * 2; if (v) *--levels = (int16_t)v, *prun++ = 15;
            v = quant[11] * 2; if (v) *--levels = (int16_t)v, *prun++ = 14;
            v = quant[14] * 2; if (v) *--levels = (int16_t)v, *prun++ = 13;
            v = quant[13] * 2; if (v) *--levels = (int16_t)v, *prun++ = 12;
            v = quant[10] * 2; if (v) *--levels = (int16_t)v, *prun++ = 11;
            v = quant[7] * 2; if (v) *--levels = (int16_t)v, *prun++ = 10;
            v = quant[3] * 2; if (v) *--levels = (int16_t)v, *prun++ = 9;
            v = quant[6] * 2; if (v) *--levels = (int16_t)v, *prun++ = 8;
            v = quant[9] * 2; if (v) *--levels = (int16_t)v, *prun++ = 7;
            v = quant[12] * 2; if (v) *--levels = (int16_t)v, *prun++ = 6;
            v = quant[8] * 2; if (v) *--levels = (int16_t)v, *prun++ = 5;
            v = quant[5] * 2; if (v) *--levels = (int16_t)v, *prun++ = 4;
            v = quant[2] * 2; if (v) *--levels = (int16_t)v, *prun++ = 3;
            v = quant[1] * 2; if (v) *--levels = (int16_t)v, *prun++ = 2;
            v = quant[4] * 2; if (v) *--levels = (int16_t)v, *prun++ = 1;
        }
    }
    else
    {
        int v;
        v = quant[3] * 2; if (v) *--levels = (int16_t)v, *prun++ = 4;
        v = quant[2] * 2; if (v) *--levels = (int16_t)v, *prun++ = 3;
        v = quant[1] * 2; if (v) *--levels = (int16_t)v, *prun++ = 2;
        v = quant[0] * 2; if (v) *--levels = (int16_t)v, *prun++ = 1;
    }
    quant = zzquant + ((maxNumCoeff == 4) ? 4 : 16);
    nnz = (int)(quant - levels);
#else
        quant += (maxNumCoeff == 4) ? 4 : 16;
    levels = quant;
    do
    {
        int v = *--quant;
        if (v)
        {
            *--levels = v * 2;
            *prun++ = cloop;
        }
    } while (--cloop);
    quant += maxNumCoeff;
    nnz = quant - levels;
#endif

    if (nnz)
    {
        cloop = MIN(3, nnz);
        levels = quant - 1;
        do
        {
            if ((unsigned)(*levels + 2) > 4u)
            {
                break;
            }
            trailing_ones_sign = (trailing_ones_sign << 1) | (*levels-- < 0);
            trailing_ones++;
        } while (--cloop);
    }
    nlevels = nnz - trailing_ones;

    nnz_context = nz_ctx[-1] + nz_ctx[1];

    nz_ctx[0] = (uint8_t)nnz;
    if (nnz_context <= 34)
    {
        nnz_context = (nnz_context + 1) >> 1;
    }
    nnz_context &= 31;

    // 9.2.1 Parsing process for total number of transform coefficient levels and trailing ones
    {
        int off = h264e_g_coeff_token[nnz_context];
        int n = 6, val = h264e_g_coeff_token[off + trailing_ones + 4 * nlevels];
        if (off != 230)
        {
            n = (val & 15) + 1;
            val >>= 4;
        }
        BS_PUT(n, val);
    }

    if (nnz)
    {
        if (trailing_ones)
        {
            BS_PUT(trailing_ones, trailing_ones_sign);
        }
        if (nlevels)
        {
            int vlcnum = 1;
            int sym_len, prefix_len;

            int sym = *levels-- - 2;
            if (sym < 0) sym = -3 - sym;
            if (sym >= 6) vlcnum++;
            if (trailing_ones < 3)
            {
                sym -= 2;
                if (nnz > 10)
                {
                    sym_len = 1;
                    prefix_len = sym >> 1;
                    if (prefix_len >= 15)
                    {
                        // or vlcnum = 1;  goto escape;
                        prefix_len = 15;
                        sym_len = 12;
                    }
                    sym -= prefix_len << 1;
                    // bypass vlcnum advance due to sym -= 2; above
                    goto loop_enter;
                }
            }

            if (sym < 14)
            {
                prefix_len = sym;
                sym = 0; // to avoid side effect in bitbuf
                sym_len = 0;
            }
            else if (sym < 30)
            {
                prefix_len = 14;
                sym_len = 4;
                sym -= 14;
            }
            else
            {
                vlcnum = 1;
                goto escape;
            }
            goto loop_enter;

            for (;;)
            {
                sym_len = vlcnum;
                prefix_len = sym >> vlcnum;
                if (prefix_len >= 15)
                {
                escape:
                    prefix_len = 15;
                    sym_len = 12;
                }
                sym -= prefix_len << vlcnum;

                if (prefix_len >= 3 && vlcnum < 6)
                    vlcnum++;
            loop_enter:
                sym |= 1 << sym_len;
                sym_len += prefix_len + 1;
                BS_PUT(sym_len, sym);
                if (!--nlevels) break;
                sym = *levels-- - 2;
                if (sym < 0) sym = -3 - sym;
            }
        }

        if (nnz < maxNumCoeff)
        {
            const uint8_t* vlc = (maxNumCoeff == 4) ? h264e_g_total_zeros_cr_2x2 : h264e_g_total_zeros;
            uint8_t* run = runs;
            int run_prev = *run++;
            int nzeros = run_prev - nnz;
            int zeros_left = 2 * nzeros - 1;
            int ctx = nnz - 1;
            run[nnz - 1] = (uint8_t)maxNumCoeff; // terminator
            for (;;)
            {
                int t;

                int val = vlc[vlc[ctx] + nzeros];
                int n = val & 15;
                val >>= 4;
                BS_PUT(n, val);

                zeros_left -= nzeros;
                if (zeros_left < 0)
                {
                    break;
                }

                t = *run++;
                nzeros = run_prev - t - 1;
                if (nzeros < 0)
                {
                    break;
                }
                run_prev = t;
                assert(zeros_left < 14);
                vlc = h264e_g_run_before;
                ctx = zeros_left;
            }
        }
    }
    BS_CLOSE(bs);
}
#endif /* H264E_ENABLE_PLAIN_C || (H264E_ENABLE_NEON && !defined(MINIH264_ASM)) */

/************************************************************************/
/*      Constants (can't be changed)                                    */
/************************************************************************/

#define MIN_QP          10   // Minimum QP

#define MVPRED_MEDIAN   1
#define MVPRED_L        2
#define MVPRED_U        3
#define MVPRED_UR       4
#define MV_NA           0x8000
#define AVAIL(mv)       ((mv).u32 != MV_NA)

#define SLICE_TYPE_P    0
#define SLICE_TYPE_I    2

#define NNZ_NA          64

#define MAX_MV_CAND     20

#define STARTCODE_4BYTES 4

/************************************************************************/
/*      Hardcoded params (can be changed at compile time)               */
/************************************************************************/
#define ALPHA_OFS       0       // Deblock alpha offset
#define BETA_OFS        0       // Deblock beta offset
#define DQP_CHROMA      0       // chroma delta QP

#define MV_RANGE        32      // Motion vector search range, pixels
#define MV_GUARD        14      // Out-of-frame MV's restriction, pixels

/************************************************************************/
/*      Code shortcuts                                                  */
/************************************************************************/
#define U(n,v) h264e_bs_put_bits(enc->bs, n, v)
#define U1(v)  h264e_bs_put_bits(enc->bs, 1, v)
#define UE(v)  h264e_bs_put_golomb(enc->bs, v)
#define SE(v)  h264e_bs_put_sgolomb(enc->bs, v)
#define SWAP(datatype, a, b) { datatype _ = a; a = b; b = _; }
#define SQR(x) ((x)*(x))
#define SQRP(pnt) SQR(pnt.s.x) + SQR(pnt.s.y)
#define SMOOTH(smth, p) smth.s.x = (63*smth.s.x + p.s.x + 32) >> 6;  smth.s.y = (63*smth.s.y + p.s.y + 32) >> 6;
#define MUL_LAMBDA(x, lambda) ((x)*(lambda) >> 4)


/************************************************************************/
/*      Arithmetics                                                     */
/************************************************************************/

#ifndef __arm__
/**
*   Count of leading zeroes
*/
static unsigned __clz(unsigned v)
{
#if defined(_MSC_VER)
    unsigned long nbit;
    _BitScanReverse(&nbit, v);
    return 31 - nbit;
#elif defined(__GNUC__) || defined(__clang__) || defined(__aarch64__)
    return __builtin_clz(v);
#else
    unsigned clz = 32;
    assert(v);
    do
    {
        clz--;
    } while (v >>= 1);
    return clz;
#endif
}
#endif

/**
*   Size of unsigned Golomb code
*/
static int bitsize_ue(int v)
{
    return 2 * (32 - __clz(v + 1)) - 1;
}

/**
*   Size of signed Golomb code
*/
static int bits_se(int v)
{
    v = 2*v - 1;
    v ^= v >> 31;
    return bitsize_ue(v);
}


/************************************************************************/
/*      Motion Vector arithmetics                                       */
/************************************************************************/

static point_t point(int x, int y)
{
    point_t p;
    p.u32 = ((unsigned)y << 16) | ((unsigned)x & 0xFFFF);    // assumes little-endian
    return p;
}

static int mv_is_zero(point_t p)
{
    return !p.u32;
}

static int mv_equal(point_t p0, point_t p1)
{
    return (p0.u32 == p1.u32);
}

/**
*   check that difference between given MV's components is greater than 3
*/
static int mv_differs3(point_t p0, point_t p1)
{
    return ABS(p0.s.x - p1.s.x) > 3 || ABS(p0.s.y - p1.s.y) > 3;
}

static point_t mv_add(point_t a, point_t b)
{
#if defined(__arm__)
    a.u32 = __sadd16(a.u32, b.u32);
#elif H264E_ENABLE_SSE2 && (H264E_CONFIGS_COUNT == 1)
    a.u32 = _mm_cvtsi128_si32(_mm_add_epi16(_mm_cvtsi32_si128(a.u32), _mm_cvtsi32_si128(b.u32)));
#else
    a.s.x += b.s.x;
    a.s.y += b.s.y;
#endif
    return a;
}

static point_t mv_sub(point_t a, point_t b)
{
#if defined(__arm__)
    a.u32 = __ssub16(a.u32, b.u32);
#elif H264E_ENABLE_SSE2 && (H264E_CONFIGS_COUNT == 1)
    a.u32 = _mm_cvtsi128_si32(_mm_sub_epi16(_mm_cvtsi32_si128(a.u32), _mm_cvtsi32_si128(b.u32)));
#else
    a.s.x -= b.s.x;
    a.s.y -= b.s.y;
#endif
    return a;
}

static void mv_clip(point_t* h264e_restrict p, const rectangle_t* range)
{
    p->s.x = MAX(p->s.x, range->tl.s.x);
    p->s.x = MIN(p->s.x, range->br.s.x);
    p->s.y = MAX(p->s.y, range->tl.s.y);
    p->s.y = MIN(p->s.y, range->br.s.y);
}

static int mv_in_rect(point_t p, const rectangle_t* r)
{
    return (p.s.y >= r->tl.s.y && p.s.y <= r->br.s.y && p.s.x >= r->tl.s.x && p.s.x <= r->br.s.x);
}

static point_t mv_round_qpel(point_t p)
{
    return point((p.s.x + 1) & ~3, (p.s.y + 1) & ~3);
}

/************************************************************************/
/*      Misc macroblock helper functions                                */
/************************************************************************/
/**
*   @return current macroblock input luma pixels
*/
static pix_t* mb_input_luma(h264e_enc_t* enc)
{
    return enc->inp.yuv[0] + (enc->mb.x + enc->mb.y * enc->inp.stride[0]) * 16;
}

/**
*   @return current macroblock input chroma pixels
*/
static pix_t* mb_input_chroma(h264e_enc_t* enc, int uv)
{
    return enc->inp.yuv[uv] + (enc->mb.x + enc->mb.y * enc->inp.stride[uv]) * 8;
}

/**
*   @return absolute MV for current macroblock for given MV
*/
static point_t mb_abs_mv(h264e_enc_t *enc, point_t mv)
{
    return mv_add(mv, point(enc->mb.x*64, enc->mb.y*64));
}

/************************************************************************/
/*      Pixel copy functions                                            */
/************************************************************************/
/**
*   Copy incomplete (cropped) macroblock pixels with borders extension
*/
static void pix_copy_cropped_mb(pix_t* d, int d_stride, const pix_t* s, int s_stride, int w, int h)
{
    int nbottom = d_stride - h; // assume dst = square d_strideXd_stride
    s_stride -= w;
    do
    {
        int cloop = w;
        pix_t last = 0;
        do
        {
            last = *s++;
            *d++ = last;
        } while (--cloop);
        cloop = d_stride - w;
        if (cloop) do
        {
            *d++ = last;    // extend row
        } while (--cloop);
        s += s_stride;
    } while (--h);
    s = d - d_stride;
    if (nbottom) do
    {
        memcpy(d, s, d_stride);  // extend columns
        d += d_stride;
    } while (--nbottom);
}

/**
*   Extend decoded picture with borders
*/
static void dec_add_border(h264e_enc_t *enc)
{
    int c, h = enc->frame.h, w = enc->frame.w, guard = 16;
    for (c = 0; c < 3; c++)
    {
       h264e_copy_borders(enc->dec.yuv[c], w, h, enc->dec.stride[c], guard);
       if (!c) guard >>= 1, w >>= 1, h >>= 1;
    }
}

/************************************************************************/
/*      Median MV predictor                                             */
/************************************************************************/

/**
*   @return neighbors availability flags for current macroblock
*/
static int mb_avail_flag(const h264e_enc_t* enc)
{
    int nmb = enc->mb.num;
    int flag = nmb >= enc->slice.start_mb_num + enc->frame.nmbx;
    if (nmb >= enc->slice.start_mb_num + enc->frame.nmbx - 1 && enc->mb.x != enc->frame.nmbx - 1)
    {
        flag += AVAIL_TR;
    }
    if (nmb != enc->slice.start_mb_num && enc->mb.x)
    {
        flag += AVAIL_L;
    }
    if (nmb > enc->slice.start_mb_num + enc->frame.nmbx && enc->mb.x)
    {
        flag += AVAIL_TL;
    }
    return flag;
}

/**
*   @return median of 3 given integers
*/
#if !(H264E_ENABLE_SSE2 && (H264E_CONFIGS_COUNT == 1))
static int me_median_of_3(int a, int b, int c)
{
    return MAX(MIN(MAX(a, b), c), MIN(a, b));
}
#endif

/**
*   @return median of 3 given motion vectors
*/
static point_t point_median_of_3(point_t a, point_t b, point_t c)
{
#if H264E_ENABLE_SSE2 && (H264E_CONFIGS_COUNT == 1)
    __m128i a2 = _mm_cvtsi32_si128(a.u32);
    __m128i b2 = _mm_cvtsi32_si128(b.u32);
    point_t med;
    med.u32 = _mm_cvtsi128_si32(_mm_max_epi16(_mm_min_epi16(_mm_max_epi16(a2, b2), _mm_cvtsi32_si128(c.u32)), _mm_min_epi16(a2, b2)));
    return med;
#else
    return point(me_median_of_3(a.s.x, b.s.x, c.s.x),
                 me_median_of_3(a.s.y, b.s.y, c.s.y));
#endif
}

/**
*   Save state of the MV predictor
*/
static void me_mv_medianpredictor_save_ctx(h264e_enc_t *enc, point_t *ctx)
{
    int i;
    point_t *mvtop = enc->mv_pred + 8 + enc->mb.x*4;
    for (i = 0; i < 4; i++)
    {
        *ctx++ = enc->mv_pred[i];
        *ctx++ = enc->mv_pred[4 + i];
        *ctx++ = mvtop[i];
    }
}

/**
*   Restore state of the MV predictor
*/
static void me_mv_medianpredictor_restore_ctx(h264e_enc_t *enc, const point_t *ctx)
{
    int i;
    point_t *mvtop = enc->mv_pred + 8 + enc->mb.x*4;
    for (i = 0; i < 4; i++)
    {
        enc->mv_pred[i] = *ctx++;
        enc->mv_pred[4 + i] = *ctx++;
        mvtop[i] = *ctx++;
    }
}

/**
*   Put motion vector to the deblock filter matrix.
*   x,y,w,h refers to 4x4 blocks within 16x16 macroblock, and should be in the range [0,4]
*/
static void me_mv_dfmatrix_put(point_t *dfmv, int x, int y, int w, int h, point_t mv)
{
    int i;
    assert(y < 4 && x < 4);

    dfmv += y*5 + x + 5;   // 5x5 matrix without left-top cell
    do
    {
        for (i = 0; i < w; i++)
        {
            dfmv[i] = mv;
        }
        dfmv += 5;
    } while (--h);
}

/**
*   Use given motion vector for prediction
*/
static void me_mv_medianpredictor_put(h264e_enc_t* enc, int x, int y, int w, int h, point_t mv)
{
    int i;
    point_t* mvtop = enc->mv_pred + 8 + enc->mb.x * 4;
    assert(y < 4 && x < 4);

    enc->mv_pred[4 + y] = mvtop[x + w - 1]; // top-left corner = top-right corner
    for (i = 1; i < h; i++)
    {
        enc->mv_pred[4 + y + i] = mv;     // top-left corner(s) for next row(s) = this
    }
    for (i = 0; i < h; i++)
    {
        enc->mv_pred[y + i] = mv;         // left = this
    }
    for (i = 0; i < w; i++)
    {
        mvtop[x + i] = mv;                // top = this
    }
}

/**
*   Motion vector median predictor for non-skip macroblock, as defined in the standard
*/
static point_t me_mv_medianpredictor_get(const h264e_enc_t *enc, point_t xy, point_t wh)
{
    int x = xy.s.x >> 2;
    int y = xy.s.y >> 2;
    int w = wh.s.x >> 2;
    int h = wh.s.y >> 2;
    int mvPredType = MVPRED_MEDIAN;
    point_t a, b, c, d, ret = point(0, 0);
    point_t *mvtop = enc->mv_pred + 8 + enc->mb.x*4;
    int flag = enc->mb.avail;

    assert(y < 4);
    assert(x < 4);
    assert(w <= 4);
    assert(h <= 4);

    a = enc->mv_pred[y];
    b = mvtop[x];
    c = mvtop[x + w];
    d = enc->mv_pred[4 + y];

    if (!x)
    {
        if (!(flag & AVAIL_L))
        {
            a.u32 = MV_NA;
        }
        if (!(flag & AVAIL_TL))
        {
            d.u32 = MV_NA;
        }
    }
    if (!y)
    {
        if (!(flag & AVAIL_T))
        {
            b.u32 = MV_NA;
            if (x + w < 4)
            {
                c.u32 = MV_NA;
            }
            if (x > 0)
            {
                d.u32 = MV_NA;
            }
        }
        if (!(flag & AVAIL_TL) && !x)
        {
            d.u32 = MV_NA;
        }
        if (!(flag & AVAIL_TR) && x + w == 4)
        {
            c.u32 = MV_NA;
        }
    }

    if (x + w == 4 && (!(flag & AVAIL_TR) || y))
    {
        c = d;
    }

    if (AVAIL(a) && !AVAIL(b) && !AVAIL(c))
    {
        mvPredType = MVPRED_L;
    } else if (!AVAIL(a) && AVAIL(b) && !AVAIL(c))
    {
        mvPredType = MVPRED_U;
    } else if (!AVAIL(a) && !AVAIL(b) && AVAIL(c))
    {
        mvPredType = MVPRED_UR;
    }

    // Directional predictions
    if (w == 2 && h == 4)
    {
        if (x == 0)
        {
            if (AVAIL(a))
            {
                mvPredType = MVPRED_L;
            }
        } else
        {
            if (AVAIL(c))
            {
                mvPredType = MVPRED_UR;
            }
        }
    } else if (w == 4 && h == 2)
    {
        if (y == 0)
        {
            if (AVAIL(b))
            {
                mvPredType = MVPRED_U;
            }
        } else
        {
            if (AVAIL(a))
            {
                mvPredType = MVPRED_L;
            }
        }
    }

    switch(mvPredType)
    {
    default:
    case MVPRED_MEDIAN:
        if (!(AVAIL(b) || AVAIL(c)))
        {
            if (AVAIL(a))
            {
                ret = a;
            }
        } else
        {
            if (!AVAIL(a))
            {
                a = ret;
            }
            if (!AVAIL(b))
            {
                b = ret;
            }
            if (!AVAIL(c))
            {
                c = ret;
            }
            ret = point_median_of_3(a, b, c);
        }
        break;
    case MVPRED_L:
        if (AVAIL(a))
        {
            ret = a;
        }
        break;
    case MVPRED_U:
        if (AVAIL(b))
        {
            ret = b;
        }
        break;
    case MVPRED_UR:
        if (AVAIL(c))
        {
            ret = c;
        }
        break;
    }
    return ret;
}

/**
*   Motion vector median predictor for skip macroblock
*/
static point_t me_mv_medianpredictor_get_skip(h264e_enc_t *enc)
{
    point_t pred_16x16 = me_mv_medianpredictor_get(enc, point(0, 0),  point(16, 16));
    enc->mb.mv_skip_pred = point(0, 0);
    if (!(~enc->mb.avail & (AVAIL_L | AVAIL_T)))
    {
        point_t *mvtop = enc->mv_pred + 8 + enc->mb.x*4;
        if (!mv_is_zero(enc->mv_pred[0]) && !mv_is_zero(mvtop[0]))
        {
            enc->mb.mv_skip_pred = pred_16x16;
        }
    }
    return pred_16x16;
}

/**
*   Get starting points candidates for MV search
*/
static int me_mv_medianpredictor_get_cand(const h264e_enc_t *enc, point_t *mv)
{
    point_t *mv0 = mv;
    point_t *mvtop = enc->mv_pred + 8 + enc->mb.x*4;
    int flag = enc->mb.avail;
    *mv++ = point(0, 0);
    if ((flag & AVAIL_L) && AVAIL(enc->mv_pred[0]))
    {
        *mv++ = enc->mv_pred[0];
    }
    if ((flag & AVAIL_T) && AVAIL(mvtop[0]))
    {
        *mv++ = mvtop[0];
    }
    if ((flag & AVAIL_TR) && AVAIL(mvtop[4]))
    {
        *mv++ = mvtop[4];
    }
    return (int)(mv - mv0);
}

/************************************************************************/
/*      NAL encoding                                                    */
/************************************************************************/

/**
*   Count ## of escapes, i.e. binary strings 0000 0000  0000 0000  0000 00xx
*   P(escape) = 2^-22
*   E(run_between_escapes) = 2^21 ~= 2 MB
*/
static int nal_count_esc(const uint8_t* s, int n)
{
    int i, cnt_esc = 0, cntz = 0;
    for (i = 0; i < n; i++)
    {
        uint8_t byte = *s++;
        if (cntz == 2 && byte <= 3)
        {
            cnt_esc++;
            cntz = 0;
        }

        if (byte)
        {
            cntz = 0;
        }
        else
        {
            cntz++;
        }
    }
    return cnt_esc;
}

/**
*   Put NAL escape codes to the output bitstream
*/
static int nal_put_esc(uint8_t* d, const uint8_t* s, int n)
{
    int i, j = 0, cntz = 0;
    for (i = 0; i < n; i++)
    {
        uint8_t byte = *s++;
        if (cntz == 2 && byte <= 3)
        {
            d[j++] = 3;
            cntz = 0;
        }

        if (byte)
        {
            cntz = 0;
        }
        else
        {
            cntz++;
        }
        d[j++] = byte;
    }
    assert(d + j <= s);
    return j;
}

/**
*   Init NAL encoding
*/
static void nal_start(h264e_enc_t* enc, int nal_hdr)
{
    uint8_t* d = enc->out + enc->out_pos;
    d[0] = d[1] = d[2] = 0; d[3] = 1; // start code
    enc->out_pos += STARTCODE_4BYTES;
    d += STARTCODE_4BYTES + (-(int)enc->out_pos & 3);   // 4-bytes align for bitbuffer
    assert(IS_ALIGNED(d, 4));
    h264e_bs_init_bits(enc->bs, d);
    U(8, nal_hdr);
}

/**
*   Finalize NAL encoding
*/
static void nal_end(h264e_enc_t* enc)
{
    int cnt_esc, bs_bytes;
    uint8_t* nal = enc->out + enc->out_pos;

    U1(1); // stop bit
    bs_bytes = h264e_bs_byte_align(enc->bs) >> 3;
    h264e_bs_flush(enc->bs);

    // count # of escape bytes to insert
    cnt_esc = nal_count_esc((unsigned char*)enc->bs->origin, bs_bytes);

    if ((uint8_t*)enc->bs->origin != nal + cnt_esc)
    {
        // make free space for escapes and remove align bytes
        memmove(nal + cnt_esc, enc->bs->origin, bs_bytes);
    }
    if (cnt_esc)
    {
        // insert escape bytes
        bs_bytes = nal_put_esc(nal, nal + cnt_esc, bs_bytes);
    }
    enc->out_pos += bs_bytes;
}


/************************************************************************/
/*      Top-level syntax elements (SPS,PPS,Slice)                       */
/************************************************************************/

//temp global
//#define dependency_id 1
#define quality_id 0
//#define default_base_mode_flag 0
// TODO: use log2_max_frame_num_minus4 from profile
#define log2_max_frame_num_minus4 1

/**
*   Encode Sequence Parameter Set (SPS)
*   ref: [1] 7.3.2.1.1
*/
/*
static void encode_sps(h264e_enc_t* enc, int profile_idc, int sps_id)
{
    struct limit_t
    {
        uint8_t level;
        uint8_t constrains;
        uint16_t max_fs;
        uint16_t max_vbvdiv5;
        uint32_t max_dpb;
    };
    static const struct limit_t limit[] = {
        {10, 0xE0, 99,    175 / 5, 396},
        {10, 0xF0, 99,    350 / 5, 396},
        {11, 0xE0, 396,   500 / 5, 900},
        {12, 0xE0, 396,   1000 / 5, 2376},
        {13, 0xE0, 396,   2000 / 5, 2376},
        {20, 0xE0, 396,   2000 / 5, 2376},
        {21, 0xE0, 792,   4000 / 5, 4752},
        {22, 0xE0, 1620,  4000 / 5, 8100},
        {30, 0xE0, 1620,  10000 / 5, 8100},
        {31, 0xE0, 3600,  14000 / 5, 18000},
        {32, 0xE0, 5120,  20000 / 5, 20480},
        {40, 0xE0, 8192,  25000 / 5, 32768},
        {41, 0xE0, 8192,  62500 / 5, 32768},
        {42, 0xE0, 8704,  62500 / 5, 34816},
        {50, 0xE0, 22080, 135000 / 5, 110400},
        {51, 0xE0, 36864, 240000 / 5, 184320}
    };
    const struct limit_t* plim = limit;

    const int max_num_ref_frames = 0;
    const int vbv_size_bytes = 0;

    while (plim->level < 51 && (enc->frame.nmb > plim->max_fs ||
        vbv_size_bytes > plim->max_vbvdiv5 * (5 * 1000 / 8) ||
        (unsigned)(enc->frame.nmb * max_num_ref_frames) > plim->max_dpb))
    {
        plim++;
    }

    nal_start(enc, 0x67);
    U(8, profile_idc);  // profile, 66 = baseline
    U(8, plim->constrains & 4);     // no constrains
    U(8, plim->level);
    UE(sps_id);

    UE(log2_max_frame_num_minus4);  // log2_max_frame_num_minus4  1 UE(0);  // log2_max_frame_num_minus4  1
    UE(2);  // pic_order_cnt_type         011
    UE(max_num_ref_frames);  // num_ref_frames
    U1(0);                                      // gaps_in_frame_num_value_allowed_flag);
    UE(((enc->param.width + 15) >> 4) - 1);     // pic_width_in_mbs_minus1
    UE(((enc->param.height + 15) >> 4) - 1);    // pic_height_in_map_units_minus1
    U(3, 6 + enc->frame.cropping_flag);         // frame_mbs_only_flag|direct_8x8_inference_flag|frame_cropping_flag
    //    U1(1);  // frame_mbs_only_flag
    //    U1(1);  // direct_8x8_inference_flag
    //    U1(frame_cropping_flag);  // frame_cropping_flag
    if (enc->frame.cropping_flag)
    {
        UE(0);                                          // frame_crop_left_offset
        UE((enc->frame.w - enc->param.width) >> 1);     // frame_crop_right_offset
        UE(0);                                          // frame_crop_top_offset
        UE((enc->frame.h - enc->param.height) >> 1);    // frame_crop_bottom_offset
    }
    U1(0);      // vui_parameters_present_flag

    nal_end(enc);
}
*/

/**
*   Encode Picture Parameter Set (SPS)
*   ref: [1] 7.3.2.2
*/
/*
static void encode_pps(h264e_enc_t* enc, int sps_id, int pps_id)
{
    nal_start(enc, 0x68);
    //   U(10, 0x338);       // constant shortcut:
    UE(pps_id);  // pic_parameter_set_id         1
    UE(sps_id);  // seq_parameter_set_id         1
    U1(0);  // entropy_coding_mode_flag     0
    U1(0);  // pic_order_present_flag       0
    UE(0);  // num_slice_groups_minus1      1
    UE(0);  // num_ref_idx_l0_active_minus1 1
    UE(0);  // num_ref_idx_l1_active_minus1 1
    U1(0);  // weighted_pred_flag           0
    U(2, 0); // weighted_bipred_idc          00
    SE(enc->sps.pic_init_qp - 26);  // pic_init_qp_minus26
#if DQP_CHROMA
    SE(0);  // pic_init_qs_minus26                    1
    SE(DQP_CHROMA);  // chroma_qp_index_offset        1
    U1(1);  // deblocking_filter_control_present_flag 1
    U1(0);  // constrained_intra_pred_flag            0
    U1(0);  // redundant_pic_cnt_present_flag         0
#else
    U(5, 0x1C);         // constant shortcut:
    //     SE(0);  // pic_init_qs_minus26                    1
    //     SE(0);  // chroma_qp_index_offset                 1
    //     U1(1);  // deblocking_filter_control_present_flag 1
    //     U1(0);  // constrained_intra_pred_flag            0
    //     U1(0);  // redundant_pic_cnt_present_flag         0
#endif
    nal_end(enc);
}
*/

/**
*   Encode Slice Header
*   ref: [1] 7.3.3
*/
static void encode_slice_header(h264e_enc_t* enc, int frame_type, int pps_id)
{
    // slice reset
    enc->slice.start_mb_num = enc->mb.num;
    enc->mb.skip_run = 0;
    memset(enc->i4x4mode, -1, (enc->frame.nmbx + 1)*4);
    memset(enc->nnz, NNZ_NA, (enc->frame.nmbx + 1) * 8);    // DF ignore slice borders, but uses it's own nnz's

    //const int is_reference_frame = frame_type == H264E_FRAME_TYPE_KEY; // nal_ref_idc != 0
    // TODO: take from passed slice header; for now every frame is a reference frame
    const int is_reference_frame = 1; // nal_ref_idc != 0
    nal_start(enc, (frame_type == H264E_FRAME_TYPE_KEY ? 0x5 : 1) | (is_reference_frame ? 0x60 : 0));

    UE(enc->slice.start_mb_num);        // first_mb_in_slice
    UE(enc->slice.type);                // slice_type
    UE(pps_id);                           // pic_parameter_set_id
    U(4 + log2_max_frame_num_minus4, enc->frame.num & ((1 << (log2_max_frame_num_minus4 + 4)) - 1)); // frame_num U(4, enc->frame.num&15);            // frame_num
    if (frame_type == H264E_FRAME_TYPE_KEY)
    {
        UE(enc->next_idr_pic_id);       // idr_pic_id
    }
    if (enc->slice.type == SLICE_TYPE_P) {
        U1(0);  // num_ref_idx_active_override_flag
        // ref_pic_list_modification():
        U1(0);  //  ref_pic_list_modification_flag_l0
    }

    // dec_ref_pic_marking():
    if (is_reference_frame) {
        if (frame_type == H264E_FRAME_TYPE_KEY) {
            U1(0);  // no_output_of_prior_pics_flag
            U1(0);  // long_term_reference_flag
        }
        else {
            // 7.3.3.3 Decoded reference picture marking syntax
            U1(0);  // adaptive_ref_pic_marking_mode_flag
        }
    }
    
    SE(enc->rc.prev_qp - enc->sps.pic_init_qp);     // slice_qp_delta
    UE(enc->speed.disable_deblock);             // disable deblock

    if (enc->speed.disable_deblock != 1)
    {
#if ALPHA_OFS || BETA_OFS
        SE(ALPHA_OFS/2);                            // slice_alpha_c0_offset_div2
        SE(BETA_OFS/2);                             // slice_beta_offset_div2
#else
        U(2, 3);
#endif
    }
}

/**
*   Macroblock transform, quantization and bitstream encoding
*/
static void mb_write(h264e_enc_t* enc)
{
    int i, uv, mb_type, cbpc, cbpl, cbp;
    scratch_t* qv = enc->scratch;
    int mb_type_svc = enc->mb.type;
    int intra16x16_flag = mb_type_svc >= 6;
    uint8_t nz[9];
    uint8_t* nnz_top = enc->nnz + 8 + enc->mb.x * 8;
    uint8_t* nnz_left = enc->nnz;

    if (enc->mb.type != 5)
    {
        enc->i4x4mode[0] = enc->i4x4mode[enc->mb.x + 1] = 0x02020202;
    }

    enc->df.nzflag = ((enc->df.nzflag >> 4) & 0x84210) | enc->df.df_nzflag[enc->mb.x];
    for (i = 0; i < 4; i++)
    {
        nz[5 + i] = nnz_top[i];
        nnz_top[i] = 0;
        nz[3 - i] = nnz_left[i];
        nnz_left[i] = 0;
    }

l_skip:
    if (enc->mb.type == -1)
    {
        // encode skip macroblock
        assert(enc->slice.type != SLICE_TYPE_I);

        // Increment run count
        enc->mb.skip_run++;

        // Update predictors
        *(uint32_t*)(nnz_top + 4) = *(uint32_t*)(nnz_left + 4) = 0; // set chroma NNZ to 0
        me_mv_medianpredictor_put(enc, 0, 0, 4, 4, enc->mb.mv[0]);
        me_mv_dfmatrix_put(enc->df.df_mv, 0, 0, 4, 4, enc->mb.mv[0]);

        // Update reference with reconstructed pixels
        h264e_copy_16x16(enc->dec.yuv[0], enc->dec.stride[0], enc->pbest, 16);
        h264e_copy_8x8(enc->dec.yuv[1], enc->dec.stride[1], enc->ptest);
        h264e_copy_8x8(enc->dec.yuv[2], enc->dec.stride[2], enc->ptest + 8);
    }
    else
    {
        if (enc->mb.type != 5)
        {
            unsigned nz_mask;
            nz_mask = h264e_transform_sub_quant_dequant(qv->mb_pix_inp, enc->pbest, 16, intra16x16_flag ? QDQ_MODE_INTRA_16 : QDQ_MODE_INTER, qv->qy, enc->rc.qdat[0]);
            enc->scratch->nz_mask = (uint16_t)nz_mask;
            if (intra16x16_flag)
            {
                h264e_quant_luma_dc(qv->qy, qv->quant_dc, enc->rc.qdat[0]);
                nz_mask = 0xFFFF;
            }
            h264e_transform_add(enc->dec.yuv[0], enc->dec.stride[0], enc->pbest, qv->qy, 4, nz_mask << 16);
        }

        // Coded Block Pattern for luma
        cbpl = 0;
        if (enc->scratch->nz_mask & 0xCC00) cbpl |= 1;
        if (enc->scratch->nz_mask & 0x3300) cbpl |= 2;
        if (enc->scratch->nz_mask & 0x00CC) cbpl |= 4;
        if (enc->scratch->nz_mask & 0x0033) cbpl |= 8;

        // Coded Block Pattern for chroma
        cbpc = 0;
        for (uv = 1; uv < 3; uv++)
        {
            pix_t* pred = enc->ptest + (uv - 1) * 8;
            pix_t* pix_mb_uv = mb_input_chroma(enc, uv);
            int dc_flag, inp_stride = enc->inp.stride[uv];
            unsigned nz_mask;
            quant_t* pquv = (uv == 1) ? qv->qu : qv->qv;

            if (enc->frame.cropping_flag && ((enc->mb.x + 1) * 16 > enc->param.width || (enc->mb.y + 1) * 16 > enc->param.height))
            {
                pix_copy_cropped_mb(enc->scratch->mb_pix_inp, 8, pix_mb_uv, enc->inp.stride[uv],
                    MIN(8, enc->param.width / 2 - enc->mb.x * 8),
                    MIN(8, enc->param.height / 2 - enc->mb.y * 8)
                );
                pix_mb_uv = enc->scratch->mb_pix_inp;
                inp_stride = 8;
            }

            nz_mask = h264e_transform_sub_quant_dequant(pix_mb_uv, pred, inp_stride, QDQ_MODE_CHROMA, pquv, enc->rc.qdat[1]);

            if (nz_mask)
            {
                cbpc = 2;
            }

            cbpc |= dc_flag = h264e_quant_chroma_dc(pquv, uv == 1 ? qv->quant_dc_u : qv->quant_dc_v, enc->rc.qdat[1]);

            if (!(dc_flag | nz_mask))
            {
                h264e_copy_8x8(enc->dec.yuv[uv], enc->dec.stride[uv], pred);
            }
            else
            {
                if (dc_flag)
                {
                    for (i = 0; i < 4; i++)
                    {
                        if (~nz_mask & (8 >> i))
                        {
                            memset(pquv[i].dq + 1, 0, (16 - 1) * sizeof(int16_t));
                        }
                    }
                    nz_mask = 15;
                }
                h264e_transform_add(enc->dec.yuv[uv], enc->dec.stride[uv], pred, pquv, 2, nz_mask << 28);
            }
        }

        cbpc = MIN(cbpc, 2);

        // Rollback to skip
        if (!(enc->mb.type | cbpl | cbpc) && // Inter prediction, all-zero after quantization
            mv_equal(enc->mb.mv[0], enc->mb.mv_skip_pred)) // MV == MV preditor for skip
        {
            enc->mb.type = -1;
            goto l_skip;
        }

        mb_type = enc->mb.type;
        if (mb_type_svc >= 6)   // intra 16x16
        {
            if (cbpl)
            {
                cbpl = 15;
            }
            mb_type += enc->mb.i16.pred_mode_luma + cbpc * 4 + (cbpl ? 12 : 0);
        }
        if (mb_type >= 5 && enc->slice.type == SLICE_TYPE_I)    // Intra in I slice
        {
            mb_type -= 5;
        }

        if (enc->slice.type != SLICE_TYPE_I)
        {
            UE(enc->mb.skip_run);
            enc->mb.skip_run = 0;
        }

        UE(mb_type);

        if (enc->mb.type == 3) // 8x8
        {
            for (i = 0; i < 4; i++)
            {
                UE(0);
            }
            // 0 = 8x8
            // 1 = 8x4
            // 2 = 4x8
            // 3 = 4x4
        }

        if (enc->mb.type >= 5)   // intra
        {
            int pred_mode_chroma;
            if (enc->mb.type == 5)  // intra 4x4
            {
                for (i = 0; i < 16; i++)
                {
                    int m = enc->mb.i4x4_mode[decode_block_scan[i]];
                    int nbits = 4;
                    if (m < 0)
                    {
                        m = nbits = 1;
                    }
                    U(nbits, m);
                }
            }
            pred_mode_chroma = enc->mb.i16.pred_mode_luma;
            if (!(pred_mode_chroma & 1))
            {
                pred_mode_chroma ^= 2;
            }
            UE(pred_mode_chroma);
            me_mv_medianpredictor_put(enc, 0, 0, 4, 4, point(MV_NA, 0));
        }
        else
        {
            int part, x = 0, y = 0;
            int dx = (enc->mb.type & 2) ? 2 : 4;
            int dy = (enc->mb.type & 1) ? 2 : 4;
            for (part = 0;; part++)
            {
                SE(enc->mb.mvd[part].s.x);
                SE(enc->mb.mvd[part].s.y);
                me_mv_medianpredictor_put(enc, x, y, dx, dy, enc->mb.mv[part]);
                me_mv_dfmatrix_put(enc->df.df_mv, x, y, dx, dy, enc->mb.mv[part]);
                x = (x + dx) & 3;
                if (!x)
                {
                    y = (y + dy) & 3;
                    if (!y)
                    {
                        break;
                    }
                }
            }
        }
        cbp = cbpl + (cbpc << 4);

        if (mb_type_svc < 6)
        {
            // encode cbp 9.1.2 Mapping process for coded block pattern
            static const uint8_t cbp2code[2][48] = {
                {3, 29, 30, 17, 31, 18, 37,  8, 32, 38, 19,  9, 20, 10, 11,  2, 16, 33, 34, 21, 35, 22, 39,  4,
                36, 40, 23,  5, 24,  6,  7,  1, 41, 42, 43, 25, 44, 26, 46, 12, 45, 47, 27, 13, 28, 14, 15,  0},
                {0,  2,  3,  7,  4,  8, 17, 13,  5, 18,  9, 14, 10, 15, 16, 11,  1, 32, 33, 36, 34, 37, 44, 40,
                35, 45, 38, 41, 39, 42, 43, 19,  6, 24, 25, 20, 26, 21, 46, 28, 27, 47, 22, 29, 23, 30, 31, 12}
            };
            UE(cbp2code[mb_type_svc < 5][cbp]);
        }

        if (cbp || (mb_type_svc >= 6))
        {
            SE(enc->rc.qp - enc->rc.prev_qp);
            enc->rc.prev_qp = enc->rc.qp;
        }

        // *** Huffman encoding ***

        // 1. Encode Luma DC (intra 16x16 only)
        if (intra16x16_flag)
        {
            h264e_vlc_encode(enc->bs, qv->quant_dc, 16, nz + 4);
        }


        // 2. Encode luma residual (only if CBP non-zero)
        if (cbpl)
        {
            for (i = 0; i < 16; i++)
            {
                int j = decode_block_scan[i];
                if (cbp & (1 << (i >> 2)))
                {
                    uint8_t* pnz = nz + 4 + (j & 3) - (j >> 2);
                    h264e_vlc_encode(enc->bs, qv->qy[j].qv, 16 - intra16x16_flag, pnz);
                    if (*pnz)
                    {
                        enc->df.nzflag |= 1 << (5 + (j & 3) + 5 * (j >> 2));
                    }
                }
                else
                {
                    nz[4 + (j & 3) - (j >> 2)] = 0;
                }
            }
            for (i = 0; i < 4; i++)
            {
                nnz_top[i] = nz[1 + i];
                nnz_left[i] = nz[7 - i];
            }
        }

        // 2. Encode chroma
        if (cbpc)
        {
            uint8_t nzcdc[3];
            nzcdc[0] = nzcdc[2] = 17;   // dummy neighbors, indicating chroma DC
            // 2.1. Encode chroma DC
            for (uv = 1; uv < 3; uv++)
            {
                h264e_vlc_encode(enc->bs, uv == 1 ? qv->quant_dc_u : qv->quant_dc_v, 4, nzcdc + 1);
            }

            // 2.2. Encode chroma residual
            if (cbpc > 1)
            {
                for (uv = 1; uv < 3; uv++)
                {
                    uint8_t nzc[5];
                    int nnz_off = (uv == 1 ? 4 : 6);
                    quant_t* pquv = uv == 1 ? qv->qu : qv->qv;
                    for (i = 0; i < 2; i++)
                    {
                        nzc[3 + i] = nnz_top[nnz_off + i];
                        nzc[1 - i] = nnz_left[nnz_off + i];
                    }
                    for (i = 0; i < 4; i++)
                    {
                        int k = 2 + (i & 1) - (i >> 1);
                        h264e_vlc_encode(enc->bs, pquv[i].qv, 15, nzc + k);
                    }
                    for (i = 0; i < 2; i++)
                    {
                        nnz_top[nnz_off + i] = nzc[1 + i];
                        nnz_left[nnz_off + i] = nzc[3 - i];
                    }
                }
            }
        }
        if (cbpc != 2)
        {
            *(uint32_t*)(nnz_top + 4) = *(uint32_t*)(nnz_left + 4) = 0; // set chroma NNZ to 0
        }
    }

    // Save top & left lines
    for (uv = 0; uv < 3; uv++)
    {
        int off = 0, n = uv ? 8 : 16;
        pix_t* top = enc->top_line + 48 + enc->mb.x * 32;
        pix_t* left = enc->top_line;
        pix_t* mb = enc->dec.yuv[uv];

        if (uv)
        {
            off = 8 + uv * 8;
        }
        top += off;
        left += off;

        enc->top_line[32 + uv] = top[n - 1];
        for (i = 0; i < n; i++)
        {
            left[i] = mb[n - 1 + i * enc->dec.stride[uv]];
            top[i] = mb[(n - 1) * enc->dec.stride[uv] + i];
        }
    }
}

/************************************************************************/
/*      Intra mode encoding                                             */
/************************************************************************/
/**
*   Estimate cost of 4x4 intra predictor
*/
static void intra_choose_4x4(h264e_enc_t *enc)
{
    int i, n, a, nz_mask = 0, avail = mb_avail_flag(enc);
    scratch_t *qv = enc->scratch;
    pix_t *mb_dec = enc->dec.yuv[0];
    pix_t *dec = enc->ptest;
    int cost =  g_lambda_i4_q4[enc->rc.qp];// + MUL_LAMBDA(16, g_lambda_q4[enc->rc.qp]);    // 4x4 cost: at least 16 bits + penalty

    uint32_t edge_store[(3 + 16 + 1 + 16 + 4)/4 + 2]; // pad for SSE
    pix_t *edge = ((pix_t*)edge_store) + 3 + 16 + 1;
    uint32_t *edge32 = (uint32_t *)edge;              // alias
    const uint32_t *top32 = (const uint32_t*)(enc->top_line + 48 + enc->mb.x*32);
    pix_t *left = enc->top_line;

    edge[-1] = enc->top_line[32];
    for (i = 0; i < 16; i++)
    {
        edge[-2 - i] = left[i];
    }
    for (i = 0; i < 4; i++)
    {
        edge32[i] = top32[i];
    }
    edge32[4] = top32[8];

    for (n = 0; n < 16; n++)
    {
        static const uint8_t block2avail[16] = {
            0x07, 0x23, 0x23, 0x2b, 0x9b, 0x77, 0xff, 0x77, 0x9b, 0xff, 0xff, 0x77, 0x9b, 0x77, 0xff, 0x77,
        };
        pix_t *block;
        pix_t *blockin;
        int sad, mpred, mode;
        int r = n >> 2;
        int c = n & 3;
        int8_t *ctx_l = (int8_t *)enc->i4x4mode + r;
        int8_t *ctx_t = (int8_t *)enc->i4x4mode + 4 + enc->mb.x*4 + c;
        edge = ((pix_t*)edge_store) + 3 + 16 + 1 + 4*c - 4*r;

        a = avail;
        a &= block2avail[n];
        a |= block2avail[n] >> 4;

        if (!(block2avail[n] & AVAIL_TL)) // TL replace
        {
            if ((n <= 3 && (avail & AVAIL_T)) ||
                (n  > 3 && (avail & AVAIL_L)))
            {
                a |= AVAIL_TL;
            }
        }
        if (n < 3 && (avail & AVAIL_T))
        {
            a |= AVAIL_TR;
        }

        blockin = enc->scratch->mb_pix_inp + (c + r*16)*4;
        block = dec + (c + r*16)*4;

        mpred = MIN(*ctx_l, *ctx_t);
        if (mpred < 0)
        {
            mpred = 2;
        }

        sad = h264e_intra_choose_4x4(blockin, block, a, edge, mpred, MUL_LAMBDA(3, g_lambda_q4[enc->rc.qp]));
        mode = sad & 15;
        sad >>= 4;

        *ctx_l = *ctx_t = (int8_t)mode;
        if (mode == mpred)
        {
            mode = -1;
        } else if (mode > mpred)
        {
            mode--;
        }
        enc->mb.i4x4_mode[n] = (int8_t)mode;

        nz_mask <<= 1;
        if (sad > g_skip_thr_i4x4[enc->rc.qp])
        {
            //  skip transform on low SAD gains just about 2% for all-intra coding at QP40,
            //  for other QP gain is minimal, so SAD check do not used
            nz_mask |= h264e_transform_sub_quant_dequant(blockin, block, 16, QDQ_MODE_INTRA_4, qv->qy + n, enc->rc.qdat[0]);

            if (nz_mask & 1)
            {
                h264e_transform_add(block, 16, block, qv->qy + n, 1, ~0);
            }
        } else
        {
            memset((qv->qy+n), 0, sizeof(qv->qy[0]));
        }

        cost += sad;

        edge[2] = block[3];
        edge[1] = block[3 + 16];
        edge[0] = block[3 + 16*2];
        *(uint32_t*)&edge[-4] = *(uint32_t*)&block[16*3];
    }
    enc->scratch->nz_mask = (uint16_t)nz_mask;

    if (cost < enc->mb.cost)
    {
        enc->mb.cost = cost;
        enc->mb.type = 5;   // intra 4x4
        h264e_copy_16x16(mb_dec, enc->dec.stride[0], dec, 16);  // restore reference
    }
}

/**
*   Choose 16x16 prediction mode, most suitable for given gradient
*/
static int intra_estimate_16x16(pix_t* p, int s, int avail, int qp)
{
    static const uint8_t mode_i16x16_valid[8] = { 4, 5, 6, 7, 4, 5, 6, 15 };
    pix_t p00 = p[0];
    pix_t p01 = p[15];
    pix_t p10 = p[15 * s + 0];
    pix_t p11 = p[15 * s + 15];
    int v = mode_i16x16_valid[avail & (AVAIL_T + AVAIL_L + AVAIL_TL)];
    // better than above on low bitrates
    int dx = ABS(p00 - p01) + ABS(p10 - p11) + ABS(p[8 * s] - p[8 * s + 15]);
    int dy = ABS(p00 - p10) + ABS(p01 - p11) + ABS(p[8] - p[15 * s + 8]);

    if ((dx > 30 + 3 * dy && dy < (100 + 50 - qp)
        //|| (/*dx < 50 &&*/ dy <= 12)
        ) && (v & 1))
        return 0;
    else if (dy > 30 + 3 * dx && dx < (100 + 50 - qp) && (v & (1 << 1)))
        return 1;
    else
        return 2;
}

/**
*   Estimate cost of 16x16 intra predictor
*
*   for foreman@qp10
*
*   12928 - [0-3], [0]
*   12963 - [0-2], [0]
*   12868 - [0-2], [0-3]
*   12878 - [0-2], [0-2]
*   12834 - [0-3], [0-3]
*sad
*   13182
*heuristic
*   13063
*
*/
static void intra_choose_16x16(h264e_enc_t* enc, pix_t* left, pix_t* top, int avail)
{
    int sad, sad4[4];
    // heuristic mode decision
    enc->mb.i16.pred_mode_luma = intra_estimate_16x16(enc->scratch->mb_pix_inp, 16, avail, enc->rc.qp);

    // run chosen predictor
    h264e_intra_predict_16x16(enc->ptest, left, top, enc->mb.i16.pred_mode_luma);

    // coding cost
    sad = h264e_sad_mb_unlaign_8x8(enc->scratch->mb_pix_inp, 16, enc->ptest, sad4)        // SAD
        + MUL_LAMBDA(bitsize_ue(enc->mb.i16.pred_mode_luma + 1), g_lambda_q4[enc->rc.qp]) // side-info penalty
        + g_lambda_i16_q4[enc->rc.qp];                                                    // block kind penalty

    if (sad < enc->mb.cost)
    {
        enc->mb.cost = sad;
        enc->mb.type = 6;
        SWAP(pix_t*, enc->pbest, enc->ptest);
    }
}

/************************************************************************/
/*      Inter mode encoding                                             */
/************************************************************************/

/**
*   Sub-pel luma interpolation
*/
static void interpolate_luma(const pix_t *ref, int stride, point_t mv, point_t wh, pix_t *dst)
{
    ref += (mv.s.y >> 2) * stride + (mv.s.x >> 2);
    mv.u32 &= 0x000030003;
    h264e_qpel_interpolate_luma(ref, stride, dst, wh, mv);
}

/**
*   Sub-pel chroma interpolation
*/
static void interpolate_chroma(h264e_enc_t *enc, point_t mv)
{
    int i;
    for (i = 1; i < 3; i++)
    {
        point_t wh;
        int part = 0, x = 0, y = 0;
        wh.s.x = (enc->mb.type & 2) ? 4 : 8;
        wh.s.y = (enc->mb.type & 1) ? 4 : 8;
        if (enc->mb.type == -1) // skip
        {
            wh.s.x = wh.s.y = 8;
        }

        for (;;part++)
        {
            pix_t *ref;
            mv = mb_abs_mv(enc, enc->mb.mv[part]);
            ref = enc->ref.yuv[i] + ((mv.s.y >> 3) + y)*enc->ref.stride[i] + (mv.s.x >> 3) + x;
            mv.u32 &= 0x00070007;
            h264e_qpel_interpolate_chroma(ref, enc->ref.stride[i], enc->ptest + (i - 1)*8 + 16*y + x, wh, mv);
            x = (x + wh.s.x) & 7;
            if (!x)
            {
                y = (y + wh.s.y) & 7;
                if (!y)
                {
                    break;
                }
            }
        }
    }
}

/**
*   RD cost of given MV
*/
static int me_mv_cost(point_t mv, point_t mv_pred, int qp)
{
    int nb = bits_se(mv.s.x - mv_pred.s.x) + bits_se(mv.s.y - mv_pred.s.y);
    return MUL_LAMBDA(nb, g_lambda_mv_q4[qp]);
}

/**
*   RD cost of given MV candidate (TODO)
*/
#define me_mv_cand_cost me_mv_cost
//static int me_mv_cand_cost(point_t mv, point_t mv_pred, int qp)
//{
//    int nb = bits_se(mv.s.x - mv_pred.s.x) + bits_se(mv.s.y - mv_pred.s.y);
//    return MUL_LAMBDA(nb, g_lambda_mv_q4[qp]);
//}


/**
*   Modified full-pel motion search with small diamond algorithm
*   note: diamond implemented with small modifications, trading speed for precision
*/
static int me_search_diamond(h264e_enc_t *enc, const pix_t *ref, const pix_t *b, int rowbytes, point_t *mv,
    const rectangle_t *range, int qp, point_t mv_pred, int min_sad, point_t wh, pix_t *scratch, pix_t **ppbest, int store_bytes)
{
    // cache map           cache moves
    //      3              0   x->1
    //      *              1   x->0
    //  1 * x * 0          2   x->3
    //      *              3   x->2
    //      2                   ^1

    //   cache double moves:
    //           prev               prev
    //      x ->   0   ->   3   ==>   3   =>   1
    //      x ->   0   ->   2   ==>   2   =>   1
    //      x ->   0   ->   0   ==>   0   =>   1
    //      x ->   0   ->   1   - impossible
    //   prev SAD(n) is (n+4)
    //

    static const point_t dir2mv[] = {{{4, 0}},{{-4, 0}},{{0, 4}},{{0, -4}}};
    union
    {
        uint16_t cache[8];
        uint32_t cache32[4];
    } sad;

    int dir, cloop, dir_prev, cost;
    point_t v;

    assert(mv_in_rect(*mv, range));

restart:
    dir = 0;                // start gradient descend with direction dir2mv[0]
    cloop = 4;              // try 4 directions
    dir_prev = -1;          // not yet moved

    // reset SAD cache
    sad.cache32[0] = sad.cache32[1] = sad.cache32[2] = sad.cache32[3] = ~0u;

    // 1. Full-pel ME with small diamond modification:
    // center point moved immediately as soon as new minimum found
    do
    {
        assert(dir >= 0 && dir < 4);

        // Try next point. Avoid out-of-range moves
        v = mv_add(*mv, dir2mv[dir]);
        //if (mv_in_rect(v, range) && sad.cache[dir] == (uint16_t)~0u)
        if (mv_in_rect(v, range) && sad.cache[dir] == 0xffffu)
        {
            cost = h264e_sad_mb_unlaign_wh(ref + ((v.s.y*rowbytes + v.s.x) >> 2), rowbytes, b, wh);
            //cost += me_mv_cost(*mv, mv_pred, qp);
            cost += me_mv_cost(v, mv_pred, qp);
            sad.cache[dir] = (uint16_t)cost;
            if (cost < min_sad)
            {
                // This point is better than center: move this point to center and continue
                int corner = ~0;
                if (dir_prev >= 0)                      // have previous move
                {                                       // save cache point, which can be used in next iteration
                    corner = sad.cache[4 + dir];        // see "cache double moves" above
                }
                sad.cache32[2] = sad.cache32[0];        // save current cache to 'previous'
                sad.cache32[3] = sad.cache32[1];
                sad.cache32[0] = sad.cache32[1] = ~0u;  // reset current cache
                if (dir_prev >= 0)                      // but if have previous move
                {                                       // one cache point can be reused from previous iteration
                    sad.cache[dir_prev^1] = (uint16_t)corner; // see "cache double moves" above
                }
                sad.cache[dir^1] = (uint16_t)min_sad;   // previous center become a neighbor's
                dir_prev = dir;                         // save this direction
                dir--;                                  // start next iteration with the same direction
                cloop = 4 + 1;                          // and try 4 directions (+1 for do-while loop)
                *mv = v;                                // Save best point found
                min_sad = cost;                         // and it's SAD
            }
        }
        dir = (dir + 1) & 3;                            // cycle search directions
    } while(--cloop);

    // 2. Optional: Try diagonal step
    //if (1)
    {
        int primary_dir   = sad.cache[3] >= sad.cache[2] ? 2 : 3;
        int secondary_dir = sad.cache[1] >= sad.cache[0] ? 0 : 1;
        if (sad.cache[primary_dir] < sad.cache[secondary_dir])
        {
            SWAP(int, secondary_dir, primary_dir);
        }

        v = mv_add(dir2mv[secondary_dir], dir2mv[primary_dir]);
        v = mv_add(*mv, v);
        //cost = (uint16_t)~0u;
        if (mv_in_rect(v, range))
        {
            cost = h264e_sad_mb_unlaign_wh(ref + ((v.s.y*rowbytes + v.s.x) >> 2), rowbytes, b, wh);
            cost += me_mv_cost(v, mv_pred, qp);
            if (cost < min_sad)
            {
                *mv = v;//mv_add(*mv, v);
                min_sad = cost;
                goto restart;
            }
        }
    }

    interpolate_luma(ref, rowbytes, *mv, wh, scratch);    // Plain NxM copy can be used
    *ppbest = scratch;

    // 3. Fractional pel search
    if (enc->run_param.encode_speed < 9 && mv_in_rect(*mv, &enc->frame.mv_qpel_limit))
    {
        point_t vbest = *mv;
        pix_t *pbest = scratch;
        pix_t *hpel  = scratch + store_bytes;
        pix_t *hpel1 = scratch + ((store_bytes == 8) ? 256 : 2*store_bytes);
        pix_t *hpel2 = hpel1 + store_bytes;

        int i, sad_test;
        point_t primary_qpel, secondary_qpel, vdiag;

        unsigned minsad1 = sad.cache[1];
        unsigned minsad2 = sad.cache[3];
        secondary_qpel = point(-1, 0);
        primary_qpel = point(0, -1);
        if (sad.cache[3] >= sad.cache[2])
            primary_qpel = point(0, 1), minsad2 = sad.cache[2];
        if (sad.cache[1] >= sad.cache[0])
            secondary_qpel = point(1, 0), minsad1 = sad.cache[0];

        if (minsad2 > minsad1)
        {
            SWAP(point_t, secondary_qpel, primary_qpel);
        }

        //     ============> primary
        //     |00 01 02
        //     |10 11 12
        //     |20    22
        //     V
        //     secondary
        vdiag = mv_add(primary_qpel, secondary_qpel);

        for (i = 0; i < 7; i++)
        {
            pix_t *ptest;
            switch(i)
            {
            case 0:
                // 02 = interpolate primary half-pel
                v = mv_add(*mv, mv_add(primary_qpel, primary_qpel));
                interpolate_luma(ref, rowbytes, v, wh, ptest = hpel1);
                break;
            case 1:
                // 01 q-pel = (00 + 02)/2
                v = mv_add(*mv, primary_qpel);
                h264e_qpel_average_wh_align(scratch, hpel1, ptest = hpel, wh);
                break;
            case 2:
                // 20 = interpolate secondary half-pel
                v = mv_add(*mv, mv_add(secondary_qpel, secondary_qpel));
                interpolate_luma(ref, rowbytes, v, wh, ptest = hpel2);
                break;
            case 3:
                // 10 q-pel = (00 + 20)/2
                hpel  = scratch + store_bytes; if (pbest == hpel) hpel = scratch;
                v = mv_add(*mv, secondary_qpel);
                h264e_qpel_average_wh_align(scratch, hpel2, ptest = hpel, wh);
                break;
            case 4:
                // 11 q-pel = (02 + 20)/2
                hpel  = scratch + store_bytes; if (pbest == hpel) hpel = scratch;
                v = mv_add(*mv, vdiag);
                h264e_qpel_average_wh_align(hpel1, hpel2, ptest = hpel, wh);
                break;
            case 5:
                // 22 = interpolate center half-pel
                if (pbest == hpel2) hpel2 = scratch, hpel = scratch + store_bytes;
                v = mv_add(*mv, mv_add(vdiag, vdiag));
                interpolate_luma(ref, rowbytes, v, wh, ptest = hpel2);
                break;
            case 6:
            default:
                // 12 q-pel = (02 + 22)/2
                hpel  = scratch + store_bytes; if (pbest == hpel) hpel = scratch;
                v = mv_add(*mv, mv_add(primary_qpel, vdiag));
                h264e_qpel_average_wh_align(hpel2, hpel1, ptest = hpel, wh);
                break;
            }

            sad_test = h264e_sad_mb_unlaign_wh(ptest, 16, b, wh) + me_mv_cost(v, mv_pred, qp);
            if (sad_test < min_sad)
            {
                min_sad = sad_test;
                vbest = v;
                pbest = ptest;
            }
        }

        *mv = vbest;
        *ppbest = pbest;
    }
    return min_sad;
}

/**
*   Set range for MV search
*/
static void me_mv_set_range(point_t *pnt, rectangle_t *range, const rectangle_t *mv_limit, int mby)
{
    // clip start point
    rectangle_t r = *mv_limit;
    r.tl.s.y = (int16_t)(MAX(r.tl.s.y, mby - 63*4));
    r.br.s.y = (int16_t)(MIN(r.br.s.y, mby + 63*4));
    mv_clip(pnt, &r);
    range->tl = mv_add(*pnt, point(-MV_RANGE*4, -MV_RANGE*4));
    range->br = mv_add(*pnt, point(+MV_RANGE*4, +MV_RANGE*4));
    // clip search range
    mv_clip(&range->tl, &r);
    mv_clip(&range->br, &r);
}

/**
*   Remove duplicates from MV candidates list
*/
static int me_mv_refine_cand(point_t *p, int n)
{
    int i, j, k;
    p[0] = mv_round_qpel(p[0]);
    for (j = 1, k = 1; j < n; j++)
    {
        point_t mv = mv_round_qpel(p[j]);
        for (i = 0; i < k; i++)
        {
            // TODO
            //if (!mv_differs3(mv, p[i], 3*4))
            //if (!mv_differs3(mv, p[i], 1*4))
            //if (!mv_differs3(mv, p[i], 3))
            if (mv_equal(mv, p[i]))
                break;
        }
        if (i == k)
            p[k++] = mv;
    }
    return k;
}

/**
*   Choose candidates for inter MB partitioning (16x8,8x16 or 8x8),
*   using SAD's for 8x8 sub-blocks
*/
static void mb_inter_partition(/*const */int sad[4], int mode[4])
{
/*
    slope
        |[ 1  1]| _ |[ 1 -1]|
        |[-1 -1]|   |[ 1 -1]|
        indicates v/h gradient: big negative = vertical prediction; big positive = horizontal

    skew
        |[ 1  0]| _ |[ 0 -1]|
        |[ 0 -1]|   |[ 1  0]|
        indicates diagonal gradient: big negative = diagonal down right
*/
    int p00 = sad[0];
    int p01 = sad[1];
    int p10 = sad[2];
    int p11 = sad[3];
    int sum = p00 + p01 + p10 + p11;
    int slope = ABS((p00 - p10) + (p01 - p11)) - ABS((p00 - p01) + (p10 - p11));
    int skew = ABS(p11 - p00) - ABS(p10 - p01);

    if (slope >  (sum >> 4))
    {
        mode[1] = 1;    // try 8x16 partition
    }
    if (slope < -(sum >> 4))
    {
        mode[2] = 1;    // try 16x8 partition
    }
    if (ABS(skew) > (sum >> 4) && ABS(slope) <= (sum >> 4))
    {
        mode[3] = 1;    // try 8x8 partition
    }
}

/**
*   Online MV clustering to "long" and "short" clusters
*   Estimate mean "long" and "short" vectors
*/
static void mv_clusters_update(h264e_enc_t *enc, point_t mv)
{
    int mv_norm = SQRP(mv);
    int n0 = SQRP(enc->mv_clusters[0]);
    int n1 = SQRP(enc->mv_clusters[1]);
    if (mv_norm < n1)
    {
        // "short" is shorter than "long"
        SMOOTH(enc->mv_clusters[0], mv);
    }
    if (mv_norm >= n0)
    {
        // "long" is longer than "short"
        SMOOTH(enc->mv_clusters[1], mv);
    }
}

/**
*   Choose inter mode: skip/coded, ME partition, find MV
*/
static void inter_choose_mode(h264e_enc_t *enc)
{
    int prefered_modes[4] = { 1, 0, 0, 0 };
    point_t mv_skip, mv_skip_a, mv_cand[MAX_MV_CAND];
    point_t mv_pred_16x16 = me_mv_medianpredictor_get_skip(enc);
    point_t mv_best = point(MV_NA, 0); // avoid warning

    int sad, sad_skip = 0x7FFFFFFF, sad_best = 0x7FFFFFFF;
    int off, i, j = 0, ncand = 0;
    int cand_sad4[MAX_MV_CAND][4];
    const pix_t *ref_yuv = enc->ref.yuv[0];
    int ref_stride = enc->ref.stride[0];
    int mv_cand_cost_best = 0;
    mv_skip = enc->mb.mv_skip_pred;
    mv_skip_a = mb_abs_mv(enc, mv_skip);

    for (i = 0; i < 4; i++)
    {
        enc->df.df_mv[4 + 5*i].u32 = enc->mv_pred[i].u32;
        enc->df.df_mv[i].u32       = enc->mv_pred[8 + 4*enc->mb.x + i].u32;
    }

    // Try skip mode
    if (mv_in_rect(mv_skip_a, &enc->frame.mv_qpel_limit))
    {
        int *sad4 = cand_sad4[0];
        interpolate_luma(ref_yuv, ref_stride, mv_skip_a, point(16, 16), enc->ptest);
        sad_skip = h264e_sad_mb_unlaign_8x8(enc->scratch->mb_pix_inp, 16, enc->ptest, sad4);

        if (MAX(MAX(sad4[0], sad4[1]), MAX(sad4[2], sad4[3])) < g_skip_thr_inter[enc->rc.qp])
        {
            int uv, sad_uv;

            SWAP(pix_t*, enc->pbest, enc->ptest);
            enc->mb.type = -1;
            enc->mb.mv[0] = mv_skip;
            enc->mb.cost = 0;
            interpolate_chroma(enc, mv_skip_a);

            // Check that chroma SAD is not too big for the skip
            for (uv = 1; uv <= 2; uv++)
            {
                pix_t *pred = enc->ptest + (uv - 1)*8;
                pix_t *pix_mb_uv = mb_input_chroma(enc, uv);
                int inp_stride = enc->inp.stride[uv];

                if (enc->frame.cropping_flag && ((enc->mb.x + 1)*16  > enc->param.width || (enc->mb.y + 1)*16  > enc->param.height))
                {
                    // Speculative read beyond frame borders: make local copy of the macroblock.
                    // TODO: same code used in mb_write() and mb_encode()
                    pix_copy_cropped_mb(enc->scratch->mb_pix_store, 8, pix_mb_uv, enc->inp.stride[uv],
                        MIN(8, enc->param.width/2  - enc->mb.x*8),
                        MIN(8, enc->param.height/2 - enc->mb.y*8));
                    pix_mb_uv = enc->scratch->mb_pix_store;
                    inp_stride = 8;
                }

                sad_uv = h264e_sad_mb_unlaign_wh(pix_mb_uv, inp_stride, pred, point(8, 8));
                if (sad_uv >= g_skip_thr_inter[enc->rc.qp])
                {
                    break;
                }
            }
            if (uv == 3)
            {
                return;
            }
        }

        if (enc->run_param.encode_speed < 1) // enable 8x16, 16x8 and 8x8 partitions
        {
            mb_inter_partition(sad4, prefered_modes);
        }

        //sad_skip += me_mv_cost(mv_skip, mv_pred_16x16, enc->rc.qp);

        // Too big skip SAD. Use skip predictor as a diamond start point candidate
        mv_best = mv_cand[ncand++] = mv_round_qpel(mv_skip);
        if (!((mv_skip.s.x | mv_skip.s.y) & 3))
        {
            sad_best = sad_skip;//+ me_mv_cost(mv_best, mv_pred_16x16, enc->rc.qp)
            mv_cand_cost_best = me_mv_cand_cost(mv_skip, mv_pred_16x16, enc->rc.qp);
            //mv_cand_cost_best = me_mv_cand_cost(mv_skip, point(0,0), enc->rc.qp);
            j = 1;
        }
    }

    mv_cand[ncand++] = mv_pred_16x16;
    ncand += me_mv_medianpredictor_get_cand(enc, mv_cand + ncand);

    if (enc->mb.x <= 0)
    {
        mv_cand[ncand++] = point(8*4, 0);
    }
    if (enc->mb.y <= 0)
    {
        mv_cand[ncand++] = point(0, 8*4);
    }

    mv_cand[ncand++] = enc->mv_clusters[0];
    mv_cand[ncand++] = enc->mv_clusters[1];

    assert(ncand <= MAX_MV_CAND);
    ncand = me_mv_refine_cand(mv_cand, ncand);

    for (/*j = 0*/; j < ncand; j++)
    {
        point_t mv = mb_abs_mv(enc, mv_cand[j]);
        if (mv_in_rect(mv, &enc->frame.mv_limit))
        {
            int mv_cand_cost = me_mv_cand_cost(mv_cand[j], mv_pred_16x16, enc->rc.qp);

            int *sad4 = cand_sad4[j];
            off = ((mv.s.y + 0) >> 2)*ref_stride + ((mv.s.x + 0) >> 2);
            sad = h264e_sad_mb_unlaign_8x8(ref_yuv + off, ref_stride, enc->scratch->mb_pix_inp, sad4);

            if (enc->run_param.encode_speed < 1) // enable 8x16, 16x8 and 8x8 partitions
            {
                mb_inter_partition(sad4, prefered_modes);
            }

            if (sad + mv_cand_cost < sad_best + mv_cand_cost_best)
            //if (sad < sad_best)
            {
                mv_cand_cost_best = mv_cand_cost;
                sad_best = sad;
                mv_best = mv_cand[j];
            }
        }
    }

    sad_best += me_mv_cost(mv_best, mv_pred_16x16, enc->rc.qp);

    {
        int mb_type;
        point_t wh, part, mvpred_ctx[12], part_mv[4][16], part_mvd[4][16];
        pix_t *store = enc->scratch->mb_pix_store;
        pix_t *pred_best = store, *pred_test = store + 256;

#define MAX8X8_MODES 4
        me_mv_medianpredictor_save_ctx(enc, mvpred_ctx);
        enc->mb.cost = 0xffffff;
        for (mb_type = 0; mb_type < MAX8X8_MODES; mb_type++)
        {
            static const int nbits[4] = { 1, 4, 4, 12 };
            int imv = 0;
            int part_sad = MUL_LAMBDA(nbits[mb_type], g_lambda_q4[enc->rc.qp]);

            if (!prefered_modes[mb_type]) continue;

            wh.s.x = (mb_type & 2) ? 8 : 16;
            wh.s.y = (mb_type & 1) ? 8 : 16;
            part = point(0, 0);
            for (;;)
            {
                rectangle_t range;
                pix_t *diamond_out;
                point_t mv, mv_pred, mvabs = mb_abs_mv(enc, mv_best);
                me_mv_set_range(&mvabs, &range, &enc->frame.mv_limit, enc->mb.y*16*4 + part.s.y*4);

                mv_pred = me_mv_medianpredictor_get(enc, part, wh);

                if (mb_type)
                {
                    mvabs = mv_round_qpel(mb_abs_mv(enc, mv_pred));
                    me_mv_set_range(&mvabs, &range, &enc->frame.mv_limit, enc->mb.y*16*4 + part.s.y*4);
                    off = ((mvabs.s.y >> 2) + part.s.y)*ref_stride + ((mvabs.s.x >> 2) + part.s.x);
                    sad_best = h264e_sad_mb_unlaign_wh(ref_yuv + off, ref_stride, enc->scratch->mb_pix_inp + part.s.y*16 + part.s.x, wh)
                        + me_mv_cost(mvabs,
                        //mv_pred,
                        mb_abs_mv(enc, mv_pred),
                        enc->rc.qp);
                }

                part_sad += me_search_diamond(enc, ref_yuv + part.s.y*ref_stride + part.s.x,
                    enc->scratch->mb_pix_inp + part.s.y*16 + part.s.x, ref_stride, &mvabs, &range, enc->rc.qp,
                    mb_abs_mv(enc, mv_pred), sad_best, wh,
                    store, &diamond_out, mb_type ? (mb_type == 2 ? 8 : 128) : 256);

                if (!mb_type)
                {
                    pred_test = diamond_out;
                    if (pred_test < store + 2*256)
                    {
                        pred_best = (pred_test == store ? store + 256 : store);
                        store += 2*256;
                    } else
                    {
                        pred_best = (pred_test == (store + 512) ? store + 512 + 256 : store + 512);
                    }
                } else
                {
                    h264e_copy_8x8(pred_test + part.s.y*16 + part.s.x, 16, diamond_out);
                    if (mb_type < 3)
                    {
                        int part_off = (wh.s.x >> 4)*8 + (wh.s.y >> 4)*8*16;
                        h264e_copy_8x8(pred_test + part_off + part.s.y*16 + part.s.x, 16, diamond_out + part_off);
                    }
                }

                mv = mv_sub(mvabs, point(enc->mb.x*16*4, enc->mb.y*16*4));

                part_mvd[mb_type][imv] = mv_sub(mv, mv_pred);
                part_mv[mb_type][imv++] = mv;

                me_mv_medianpredictor_put(enc, part.s.x >> 2, part.s.y >> 2, wh.s.x >> 2, wh.s.y >> 2, mv);

                part.s.x = (part.s.x + wh.s.x) & 15;
                if (!part.s.x)
                {
                    part.s.y = (part.s.y + wh.s.y) & 15;
                    if (!part.s.y) break;
                }
            }

            me_mv_medianpredictor_restore_ctx(enc, mvpred_ctx);

            if (part_sad < enc->mb.cost)
            {
                SWAP(pix_t*, pred_best, pred_test);
                enc->mb.cost = part_sad;
                enc->mb.type = mb_type;
            }
        }
        enc->pbest = pred_best;
        enc->ptest = pred_test;
        memcpy(enc->mb.mv,  part_mv [enc->mb.type], 16*sizeof(point_t));
        memcpy(enc->mb.mvd, part_mvd[enc->mb.type], 16*sizeof(point_t));

        if (enc->mb.cost > sad_skip)
        {
            enc->mb.type = 0;
            enc->mb.cost = sad_skip + me_mv_cand_cost(mv_skip, mv_pred_16x16, enc->rc.qp);
            enc->mb.mv [0] = mv_skip;
            enc->mb.mvd[0] = mv_sub(mv_skip, mv_pred_16x16);

            assert(mv_in_rect(mv_skip_a, &enc->frame.mv_qpel_limit)) ;
            interpolate_luma(ref_yuv, ref_stride, mv_skip_a, point(16, 16), enc->pbest);
            interpolate_chroma(enc, mv_skip_a);
        }
    }
}

/************************************************************************/
/*      Deblock filter                                                  */
/************************************************************************/
#define MB_FLAG_SLICE_START_DEBLOCK_2 2

/**
*   Set deblock filter strength
*/
static void df_strength(deblock_filter_t *df, int mb_type, int mbx, uint8_t *strength)
{
    uint8_t *sv = strength;
    uint8_t *sh = strength + 16;
    int flag = df->nzflag;
    df->df_nzflag[mbx] = (uint8_t)(flag >> 20);
    /*
        nzflag represents macroblock and it's neighbors with 24 bit flags:
        0 1 2 3
      4 5 6 7 8
      A B C D E
      F G H I J
      K L K N O
    */
    {
        if (mb_type < 5)
        {
            int ccloop = 4;
            point_t *mv = df->df_mv;
            do
            {
                int cloop = 4;
                do
                {
                    int v = 0;
                    if (flag & 3 << 4)
                    {
                        v = 2;
                    } else if (mv_differs3(mv[4], mv[5]))
                    {
                        v = 1;
                    }
                    *sv = (uint8_t)v; sv += 4;

                    v = 0;
                    if (flag & 33)
                    {
                        v = 2;
                    } else if (mv_differs3(mv[0], mv[5]))
                    {
                        v = 1;
                    }
                    *sh++ = (uint8_t)v;

                    flag >>= 1;
                    mv++;
                } while(--cloop);
                flag >>= 1;
                sv -= 15;
                mv++;
            } while(--ccloop);
        } else
        {
            // Deblock mode #3 (intra)
            ((uint32_t*)(sv))[1] = ((uint32_t*)(sv))[2] = ((uint32_t*)(sv))[3] =             // for inner columns
            ((uint32_t*)(sh))[1] = ((uint32_t*)(sh))[2] = ((uint32_t*)(sh))[3] = 0x03030303; // for inner rows
        }
        if ((mb_type >= 5 || df->mb_type[mbx - 1] >= 5)) // speculative read
        {
            ((uint32_t*)(strength))[0] = 0x04040404;    // Deblock mode #4 (strong intra) for left column
        }
        if ((mb_type >= 5 || df->mb_type[mbx    ] >= 5))
        {
            ((uint32_t*)(strength))[4] = 0x04040404;    // Deblock mode #4 (strong intra) for top row
        }
    }
    df->mb_type[mbx] = (int8_t)mb_type;
}

/**
*   Run deblock for current macroblock
*/
static void mb_deblock(deblock_filter_t *df, int mb_type, int qp_this, int mbx, int mby, H264E_io_yuv_t *mbyuv, int IntraBLFlag)
{
    int i, cr, qp, qp_left, qp_top;
    deblock_params_t par;
    uint8_t *alpha = par.alpha; //[2*2];
    uint8_t *beta  = par.beta;  //[2*2];
    uint32_t *strength32  = par.strength32; //[4*2]; // == uint8_t strength[16*2];
    uint8_t *strength = (uint8_t *)strength32;
    uint8_t *tc0 = par.tc0; //[16*2];

    df_strength(df, mb_type, mbx, strength);
    if (!mbx || (IntraBLFlag & MB_FLAG_SLICE_START_DEBLOCK_2))
    {
        strength32[0] = 0;
    }

    if (!mby)
    {
        strength32[4] = 0;
    }

    qp_top = df->df_qp[mbx];
    qp_left = df->df_qp[mbx - 1];
    df->df_qp[mbx] = (uint8_t)qp_this;

    cr = 0;
    for (;;)
    {
        const uint8_t *lut;
        if (*((uint32_t*)strength))
        {
            qp = (qp_left + qp_this + 1) >> 1;
            lut = g_a_tc0_b[-10 + qp + ALPHA_OFS];
            alpha[0] = lut[0];
            beta[0]  = lut[4 + (BETA_OFS - ALPHA_OFS)*5];
            for (i = 0; i < 4; i++) tc0[i] = lut[strength[i]];
        }
        if (*((uint32_t*)(strength + 16)))
        {
            qp = (qp_top + qp_this + 1) >> 1;
            lut = g_a_tc0_b[-10 + qp + ALPHA_OFS];

            alpha[2]  = lut[0];
            beta[2] = lut[4 + (BETA_OFS - ALPHA_OFS)*5];
            for (i = 0; i < 4; i++) tc0[16 + i] = lut[strength[16 + i]];
        }

        lut = g_a_tc0_b[-10 + qp_this + ALPHA_OFS];
        alpha[3] = alpha[1] = lut[0];
        beta[3] = beta[1] = lut[4 + (BETA_OFS - ALPHA_OFS)*5];
        for (i = 4; i < 16; i++)
        {
            tc0[i] = lut[strength[i]];
            tc0[16 + i] = lut[strength[16 + i]];
        }
        if (cr)
        {
            int *t = (int *)tc0;
            t[1] = t[2];         // TODO: need only for OMX
            t[5] = t[6];
            i = 2;
            do
            {
                h264e_deblock_chroma(mbyuv->yuv[i], mbyuv->stride[i], &par);
            } while (--i);
            break;
        }
        h264e_deblock_luma(mbyuv->yuv[0], mbyuv->stride[0], &par);

        qp_this = qpy2qpc[qp_this + DQP_CHROMA];
        qp_left = qpy2qpc[qp_left + DQP_CHROMA];
        qp_top = qpy2qpc[qp_top + DQP_CHROMA];
        cr++;
    }
}

/************************************************************************/
/*      Macroblock encoding                                             */
/************************************************************************/
/**
*   Macroblock encoding
*/
static void mb_encode(h264e_enc_t* enc)
{
    pix_t* top = enc->top_line + 48 + enc->mb.x * 32;
    pix_t* left = enc->top_line;
    int avail = enc->mb.avail = mb_avail_flag(enc);

    if (enc->frame.cropping_flag && ((enc->mb.x + 1) * 16 > enc->param.width || (enc->mb.y + 1) * 16 > enc->param.height))
    {
        pix_copy_cropped_mb(enc->scratch->mb_pix_inp, 16, mb_input_luma(enc), enc->inp.stride[0],
            MIN(16, enc->param.width - enc->mb.x * 16),
            MIN(16, enc->param.height - enc->mb.y * 16));
    }
    else
    {
        // cache input macroblock
        h264e_copy_16x16(enc->scratch->mb_pix_inp, 16, mb_input_luma(enc), enc->inp.stride[0]);
    }

    if (!(avail & AVAIL_L)) left = NULL;
    if (!(avail & AVAIL_T)) top = NULL;

    enc->pbest = enc->scratch->mb_pix_store;
    enc->ptest = enc->pbest + 256;
    enc->mb.type = 0;   // P16x16
    enc->mb.cost = 0x7FFFFFFF;

    if (enc->slice.type == SLICE_TYPE_P)
    {
        inter_choose_mode(enc);
    }

    if (enc->mb.type >= 0)
    {
        intra_choose_16x16(enc, left, top, avail);
        if (enc->run_param.encode_speed < 2 || enc->slice.type != SLICE_TYPE_P) // enable intra4x4 on P slices
        {
            intra_choose_4x4(enc);
        }
    }

    if (enc->mb.type < 5)
    {
        mv_clusters_update(enc, enc->mb.mv[0]);
    }

    if (enc->mb.type >= 5)
    {
        pix_t* pred = enc->ptest;
        h264e_intra_predict_chroma(pred, left + 16, top + 16, enc->mb.i16.pred_mode_luma);
    }
    else
    {
        interpolate_chroma(enc, mb_abs_mv(enc, enc->mb.mv[0]));
    }

    mb_write(enc);

    if (!enc->speed.disable_deblock)
    {
        int mbx = enc->mb.x;
        int mby = enc->mb.y;
        mb_deblock(&enc->df, enc->mb.type, enc->rc.prev_qp, mbx, mby, &enc->dec, 0);
    }
}

static void encode_slice(h264e_enc_t* enc, int frame_type, int pps_id)
{
    int i, k;

    encode_slice_header(enc, frame_type, pps_id);

    // encode frame
    do
    {   // encode row
        do
        {   // encode macroblock
            mb_encode(enc);

            enc->dec.yuv[0] += 16;
            enc->dec.yuv[1] += 8;
            enc->dec.yuv[2] += 8;

            enc->mb.num++;  // before rc_mb_end
        } while (++enc->mb.x < enc->frame.nmbx);

        for (i = 0, k = 16; i < 3; i++, k = 8)
        {
            enc->dec.yuv[i] += k * (enc->dec.stride[i] - enc->frame.nmbx);
        }
        // start new row
        enc->mb.x = 0;
        *((uint32_t*)(enc->nnz)) = *((uint32_t*)(enc->nnz + 4)) = 0x01010101 * NNZ_NA; // left edge of NNZ predictor
        enc->i4x4mode[0] = -1;

    } while (++enc->mb.y < enc->frame.nmby);

    if (enc->mb.skip_run)
    {
        UE(enc->mb.skip_run);
    }

    nal_end(enc);
    for (i = 0, k = 16; i < 3; i++, k = 8)
    {
        enc->dec.yuv[i] -= k * enc->dec.stride[i] * enc->frame.nmby;
    }
}

/**
*   Verify encoder creation parameters. Return error code, or 0 if prameters
*/
static int enc_check_create_params(const H264E_create_param_t* par)
{
    if (!par)
    {
        return H264E_STATUS_BAD_ARGUMENT;   // NULL argument
    }
    if (par->width <= 0 || par->height <= 0)
    {
        return H264E_STATUS_BAD_PARAMETER;  // non-positive frame size
    }
    if ((par->width | par->height) & 1)
    {
        return H264E_STATUS_SIZE_NOT_MULTIPLE_2; // frame size must be multiple of 2
    }
    return H264E_STATUS_SUCCESS;
};

/************************************************************************/
/*      Rate-control                                                    */
/************************************************************************/

/**
*   @return zero threshold for given rounding offset
*/
static uint16_t rc_rnd2thr(int round, int q)
{
    int b, thr = 0;
    for (b = 0x8000; b; b >>= 1)
    {
        int t = (thr | b) * q;
        if (t <= 0x10000 - round)  // TODO: error: < !!!!!!!
        {
            thr |= b;
        }
    }
    return (uint16_t)thr;
}

/**
*   Set quantizer constants (deadzone and rounding) for given QP
*/
static void rc_set_qp(h264e_enc_t* enc, int qp)
{
    //qp = MIN(qp, enc->run_param.qp_max);
    //qp = MAX(qp, enc->run_param.qp_min);
    qp = enc->run_param.qp;
    qp = MIN(qp, 51);   // avoid VC2010 static analyzer warning

    if (enc->rc.qp != qp)
    {
        static const int16_t g_quant_coeff[6 * 6] =
        {
            //    0         2         1
            13107, 10, 8066, 13, 5243, 16,
            11916, 11, 7490, 14, 4660, 18,
            10082, 13, 6554, 16, 4194, 20,
             9362, 14, 5825, 18, 3647, 23,
             8192, 16, 5243, 20, 3355, 25,
             7282, 18, 4559, 23, 2893, 29
             // 0 2 0 2
             // 2 1 2 1
             // 0 2 0 2
             // 2 1 2 1
        };

        int cloop = 2;
        enc->rc.qp = qp;
        do
        {
            uint16_t* qdat0 = enc->rc.qdat[2 - cloop];
            uint16_t* qdat = enc->rc.qdat[2 - cloop];
            int qp_div6 = qp * 86 >> 9;
            int qp_mod6 = qp - qp_div6 * 6;
            const int16_t* quant_coeff = g_quant_coeff + qp_mod6 * 6; // TODO: need calculate qp%6*6
            int i = 3;

            // Quant/dequant multiplier
            do
            {
                *qdat++ = *quant_coeff++ << 1 >> qp_div6;
                *qdat++ = *quant_coeff++ << qp_div6;
            } while (--i);

            // quantizer deadzone for P & chroma
            *qdat++ = enc->slice.type == SLICE_TYPE_P ? g_rnd_inter[qp] : g_deadzonei[qp];
            // quantizer deadzone for I
            *qdat++ = g_deadzonei[qp];

            *qdat++ = g_thr_inter[qp] - 0x7fff;
            *qdat++ = g_thr_inter2[qp] - 0x7fff;

            qdat[0] = qdat[2] = rc_rnd2thr(g_thr_inter[qp] - 0x7fff, qdat0[0]);
            qdat[1] = qdat[3] =
                qdat[4] = qdat[6] = rc_rnd2thr(g_thr_inter[qp] - 0x7fff, qdat0[2]);
            qdat[5] = qdat[7] = rc_rnd2thr(g_thr_inter[qp] - 0x7fff, qdat0[4]);
            qdat += 8;
            qdat[0] = qdat[2] = rc_rnd2thr(g_thr_inter2[qp] - 0x7fff, qdat0[0]);
            qdat[1] = qdat[3] =
                qdat[4] = qdat[6] = rc_rnd2thr(g_thr_inter2[qp] - 0x7fff, qdat0[2]);
            qdat[5] = qdat[7] = rc_rnd2thr(g_thr_inter2[qp] - 0x7fff, qdat0[4]);
            qdat += 8;
            qdat[0] = qdat[2] = qdat0[0];
            qdat[1] = qdat[3] =
                qdat[4] = qdat[6] = qdat0[2];
            qdat[5] = qdat[7] = qdat0[4];
            qdat += 8;
            qdat[0] = qdat[2] = qdat0[1];
            qdat[1] = qdat[3] =
                qdat[4] = qdat[6] = qdat0[3];
            qdat[5] = qdat[7] = qdat0[5];

            qp = qpy2qpc[qp + DQP_CHROMA];
        } while (--cloop);
    }
}

static int rc_frame_start(h264e_enc_t* enc)
{
    int qp = -1;

    enc->rc.qp = 0; // force
    rc_set_qp(enc, qp);
    qp = enc->rc.qp;

    enc->rc.prev_qp = qp;

    return 0;
}

#define ALIGN_128BIT(p) (void *)((uintptr_t)(((char*)(p)) + 15) & ~(uintptr_t)15)
#define ALLOC(ptr, size) p = ALIGN_128BIT(p); if (enc) ptr = (void *)p; p += size;

/**
*   Internal allocator for persistent RAM
*/
static int enc_alloc(h264e_enc_t* enc, const H264E_create_param_t* par, unsigned char* p)
{
    unsigned char* p0 = p;
    return (int)((p - p0) + 15) & ~15u;
}

/**
*   Internal allocator for scratch RAM
*/
static int enc_alloc_scratch(h264e_enc_t* enc, const H264E_create_param_t* par, unsigned char* p)
{
    unsigned char* p0 = p;
    int nmbx = (par->width + 15) >> 4;
    int nmby = (par->height + 15) >> 4;
    ALLOC(enc->scratch, sizeof(scratch_t));
    ALLOC(enc->out, nmbx * nmby * (384 + 2 + 10) * 3 / 2);

    ALLOC(enc->nnz, nmbx * 8 + 8);
    ALLOC(enc->mv_pred, (nmbx * 4 + 8) * sizeof(point_t));
    ALLOC(enc->i4x4mode, nmbx * 4 + 4);
    ALLOC(enc->df.df_qp, nmbx);
    ALLOC(enc->df.mb_type, nmbx);
    ALLOC(enc->df.df_nzflag, nmbx);
    ALLOC(enc->top_line, nmbx * 32 + 32 + 16);
    return (int)(p - p0);
}

static int H264E_sizeof_one(const H264E_create_param_t* par, int* sizeof_persist, int* sizeof_scratch)
{
    int error = enc_check_create_params(par);
    if (!sizeof_persist || !sizeof_scratch)
    {
        error = H264E_STATUS_BAD_ARGUMENT;
    }
    if (error)
    {
        return error;
    }

    *sizeof_persist = enc_alloc(NULL, par, (void*)(uintptr_t)1) + sizeof(h264e_enc_t);
    *sizeof_scratch = enc_alloc_scratch(NULL, par, (void*)(uintptr_t)1);
    return error;
}

/**
*   Return persistent and scratch memory requirements
*   for given encoding options.
*   See header file for details.
*/
int H264E_sizeof(const H264E_create_param_t* par, int* sizeof_persist, int* sizeof_scratch)
{
    int error = H264E_sizeof_one(par, sizeof_persist, sizeof_scratch);
    return error;
}

static int H264E_init_one(h264e_enc_t* enc, const H264E_create_param_t* opt)
{
#if H264E_CONFIGS_COUNT > 1
    init_vft(opt->enableNEON);
#endif
    memset(enc, 0, sizeof(*enc));

    enc->frame.nmbx = (opt->width + 15) >> 4;
    enc->frame.nmby = (opt->height + 15) >> 4;
    enc->frame.nmb = enc->frame.nmbx * enc->frame.nmby;
    enc->frame.w = enc->frame.nmbx * 16;
    enc->frame.h = enc->frame.nmby * 16;
    enc->frame.mv_limit.tl = point(-MV_GUARD * 4, -MV_GUARD * 4);
    enc->frame.mv_qpel_limit.tl = mv_add(enc->frame.mv_limit.tl, point(4 * 4, 4 * 4));
    enc->frame.mv_limit.br = point((enc->frame.nmbx * 16 - (16 - MV_GUARD)) * 4, (enc->frame.nmby * 16 - (16 - MV_GUARD)) * 4);
    enc->frame.mv_qpel_limit.br = mv_add(enc->frame.mv_limit.br, point(-4 * 4, -4 * 4));
    enc->frame.cropping_flag = !!((opt->width | opt->height) & 15);
    enc->param = *opt;

    enc_alloc(enc, opt, (void*)(enc + 1));

    return H264E_STATUS_SUCCESS;
}

/**
*   Encoder initialization
*   See header file for details.
*/
int H264E_init(H264E_persist_t* enc, const H264E_create_param_t* opt)
{
    h264e_enc_t* enc_curr = enc;
    int ret;

    ret = H264E_init_one(enc_curr, opt);

    return ret;
}

static int H264E_encode_one(H264E_persist_t* enc, const H264E_run_param_t* opt, int frame_type, int pps_id)
{
    // slice reset
    // TODO: slice type should be coming from std slice header structure
    enc->slice.type = frame_type == H264E_FRAME_TYPE_P ? SLICE_TYPE_P : SLICE_TYPE_I;
    rc_frame_start(enc);
    //enc->slice.type = (long_term_idx_use < 0 ? SLICE_TYPE_I : SLICE_TYPE_P);
    //rc_frame_start(enc, (long_term_idx_use < 0) ? 1 : 0, is_refers_to_long_term);

    enc->mb.x = enc->mb.y = enc->mb.num = 0;

    encode_slice(enc, frame_type, pps_id);

    //rc_frame_end(enc, long_term_idx_use == -1, enc->mb.skip_run == enc->frame.nmb, is_refers_to_long_term);

    dec_add_border(enc);

    ++enc->frame.num;

    return H264E_STATUS_SUCCESS;
}

static int check_parameters_align(const H264E_create_param_t* opt, const H264E_io_yuv_t* in)
{
    int i;
    int min_align = 0;
#if H264E_ENABLE_NEON || H264E_ENABLE_SSE2
    min_align = 7;
#endif
    //if (opt->const_input_flag && opt->temporal_denoise_flag)
    //{
    //    min_align = 0;
    //}
    for (i = 0; i < 3; i++)
    {
        if (((uintptr_t)in->yuv[i]) & min_align)
        {
            return i ? H264E_STATUS_BAD_CHROMA_ALIGN : H264E_STATUS_BAD_LUMA_ALIGN;
        }
        if (in->stride[i] & min_align)
        {
            return i ? H264E_STATUS_BAD_CHROMA_STRIDE : H264E_STATUS_BAD_LUMA_STRIDE;
        }
    }
    return H264E_STATUS_SUCCESS;
}

/**
*   Top-level encode function
*   See header file for details.
*/
int H264E_encode(H264E_persist_t* enc, H264E_scratch_t* scratch, const H264E_run_param_t* opt,
    H264E_io_yuv_t* in, H264E_io_yuv_t* ref, H264E_io_yuv_t* dec, unsigned char** coded_data, int* sizeof_coded_data)
{
    int i;
    int frame_type = opt->frame_type;
    int error;

    error = check_parameters_align(&enc->param, in);
    if (error)
    {
        return error;
    }
    if (!ref) {
       if (frame_type == H264E_FRAME_TYPE_P)
           return H264E_STATUS_BAD_ARGUMENT;
    } else {
       error = check_parameters_align(&enc->param, ref);
       if (error)
       {
           return error;
       }
    }
    error = check_parameters_align(&enc->param, dec);
    if (error)
    {
        return error;
    }
    (void)i;
    i = enc_alloc_scratch(enc, &enc->param, (unsigned char*)scratch);

    enc->inp = *in;

    enc->out_pos = 0;   // reset output bitbuffer position

    if (opt)
    {
        enc->run_param = *opt;  // local copy of run-time parameters
    }
    opt = &enc->run_param;      // refer to local copy

    // silently fix invalid QP without warning
    //if (!enc->run_param.qp_max || enc->run_param.qp_max > 51)
    //{
    //    enc->run_param.qp_max = 51;
    //}
    //if (!enc->run_param.qp_min || enc->run_param.qp_min < MIN_QP)
    //{
    //    enc->run_param.qp_min = MIN_QP;
    //}
    if (enc->run_param.qp > 51)
    {
        enc->run_param.qp = 51;
    }
    if (enc->run_param.qp < MIN_QP)
    {
        enc->run_param.qp = MIN_QP;
    }

    enc->speed.disable_deblock = (opt->encode_speed == 8 || opt->encode_speed == 10);

    const int sps_id = 0;
    const int pps_id = sps_id * 4 + 0;

    if (frame_type == H264E_FRAME_TYPE_KEY)
    {
        enc->sps.pic_init_qp = enc->run_param.qp;
        enc->next_idr_pic_id ^= 1;
        enc->frame.num = 0;

        //encode_sps(enc, 66, sps_id);
        //encode_pps(enc, sps_id, pps_id);
    }

    enc->dec = *dec;
    if (ref)
    {
        enc->ref = *ref;
    }

    H264E_encode_one(enc, opt, frame_type, pps_id);

    *sizeof_coded_data = enc->out_pos;
    *coded_data = enc->out;
    return H264E_STATUS_SUCCESS;
}
