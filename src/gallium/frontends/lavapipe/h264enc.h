#pragma once

// from https://github.com/lieff/minih264

/**
*   API return error codes
*/
#define H264E_STATUS_SUCCESS                0
#define H264E_STATUS_BAD_ARGUMENT           1
#define H264E_STATUS_BAD_PARAMETER          2
#define H264E_STATUS_SIZE_NOT_MULTIPLE_2    5
#define H264E_STATUS_BAD_LUMA_ALIGN         6
#define H264E_STATUS_BAD_LUMA_STRIDE        7
#define H264E_STATUS_BAD_CHROMA_ALIGN       8
#define H264E_STATUS_BAD_CHROMA_STRIDE      9

/**
*   Frame type definitions
*/
#define H264E_FRAME_TYPE_KEY        6       // Random access point: SPS+PPS+Intra frame
#define H264E_FRAME_TYPE_I          5       // Intra frame: updates long & short-term reference
#define H264E_FRAME_TYPE_P          2       // Use and update short-term reference

/**
*   Speed preset index.
*   Currently used values are 0, 1, 8 and 9
*/
#define H264E_SPEED_SLOWEST         0       // All coding tools enabled, including denoise filter
#define H264E_SPEED_BALANCED        5
#define H264E_SPEED_FASTEST         10      // Minimum tools enabled

/**
*   Border padding size
*/
#define H264E_BORDER_PADDING        16

/**
*   Creations parameters
*/
typedef struct H264E_create_param_tag
{
    // Frame width: must be multiple of 2
    int width;

    // Frame height: must be multiple of 2
    int height;
} H264E_create_param_t;

/**
*   Run-time parameters
*/
typedef struct H264E_run_param_tag
{
    // Variable, indicating speed/quality tradeoff
    // see H264E_SPEED_*
    // 0 means best quality
    int encode_speed;

    // Frame type override: one of H264E_FRAME_TYPE_* values
    // if 0: GOP pattern defined by create_param::gop value
    int frame_type;

    // Quantizer value, 10 indicates good quality
    // range: [10; 51]
    int qp;

} H264E_run_param_t;

/**
*    Planar YUV420 descriptor
*/
typedef struct H264E_io_yuv_tag
{
    // Pointers to 3 pixel planes of YUV image
    unsigned char* yuv[3];
    // Stride for each image plane
    int stride[3];
} H264E_io_yuv_t;

typedef struct H264E_persist_tag H264E_persist_t;
typedef struct H264E_scratch_tag H264E_scratch_t;


int H264E_sizeof(const H264E_create_param_t* par, int* sizeof_persist, int* sizeof_scratch);
int H264E_init(H264E_persist_t* enc, const H264E_create_param_t* opt);
int H264E_encode(H264E_persist_t* enc, H264E_scratch_t* scratch, const H264E_run_param_t* opt,
    H264E_io_yuv_t* in, H264E_io_yuv_t* ref, H264E_io_yuv_t* dec, unsigned char** coded_data, int* sizeof_coded_data);
