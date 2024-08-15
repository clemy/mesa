#include "lvp_private.h"

#include "vk_video/vulkan_video_codecs_common.h"

/* TODO: check whole file for calling conventions VKAPI_ATTR and VKAPI_CALL */
#define LVP_MACROBLOCK_WIDTH  16
#define LVP_MACROBLOCK_HEIGHT 16
#define LVP_ENC_MAX_RATE_LAYER 1
#define NUM_H2645_REFS        16

VKAPI_ATTR VkResult VKAPI_CALL
lvp_GetPhysicalDeviceVideoCapabilitiesKHR(VkPhysicalDevice physicalDevice, const VkVideoProfileInfoKHR *pVideoProfile,
                                           VkVideoCapabilitiesKHR *pCapabilities)
{
   //VK_FROM_HANDLE(lvp_physical_device, pdev, physicalDevice);

   if (pVideoProfile->videoCodecOperation != VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR)
      return VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR;

   if (pVideoProfile->chromaSubsampling != VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR ||
       pVideoProfile->lumaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR ||
       pVideoProfile->chromaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)
      return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

   const struct VkVideoEncodeH264ProfileInfoKHR *h264_profile =
      vk_find_struct_const(pVideoProfile->pNext, VIDEO_ENCODE_H264_PROFILE_INFO_KHR);

   if (h264_profile->stdProfileIdc != STD_VIDEO_H264_PROFILE_IDC_BASELINE)
      return VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR;

   pCapabilities->flags = VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR;
   pCapabilities->pictureAccessGranularity.width = LVP_MACROBLOCK_WIDTH;
   pCapabilities->pictureAccessGranularity.height = LVP_MACROBLOCK_HEIGHT;
   pCapabilities->minCodedExtent.width = LVP_MACROBLOCK_WIDTH;
   pCapabilities->minCodedExtent.height = LVP_MACROBLOCK_HEIGHT;
   /* TODO: set useful limits */
   pCapabilities->maxCodedExtent.width = 4096;
   pCapabilities->maxCodedExtent.height = 4096;

   struct VkVideoEncodeCapabilitiesKHR *enc_caps =
      (struct VkVideoEncodeCapabilitiesKHR *)vk_find_struct(pCapabilities->pNext, VIDEO_ENCODE_CAPABILITIES_KHR);

   if (enc_caps) {
      enc_caps->flags = VK_VIDEO_ENCODE_CAPABILITY_INSUFFICIENT_BITSTREAM_BUFFER_RANGE_DETECTION_BIT_KHR;
      // TODO: support rate control beside default
      //enc_caps->rateControlModes = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR |
      //                              VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR |
      //                              VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR;
      enc_caps->rateControlModes = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;
      enc_caps->maxRateControlLayers = LVP_ENC_MAX_RATE_LAYER;
      enc_caps->maxBitrate = 1000000000;
      enc_caps->maxQualityLevels = 1;
      enc_caps->encodeInputPictureGranularity.width = 1;
      enc_caps->encodeInputPictureGranularity.height = 1;
      enc_caps->supportedEncodeFeedbackFlags = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR |
                                                VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR;
      // TODO: support VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_HAS_OVERRIDES_BIT_KHR
      // TODO: support STATUS only queries
   }
   pCapabilities->minBitstreamBufferOffsetAlignment = 16;
   pCapabilities->minBitstreamBufferSizeAlignment = 16;
   pCapabilities->maxDpbSlots = NUM_H2645_REFS + 1;
   pCapabilities->maxActiveReferencePictures = NUM_H2645_REFS;
   
   struct VkVideoEncodeH264CapabilitiesKHR *ext = (struct VkVideoEncodeH264CapabilitiesKHR *)vk_find_struct(
      pCapabilities->pNext, VIDEO_ENCODE_H264_CAPABILITIES_KHR);

   if (ext) {
      ext->flags = VK_VIDEO_ENCODE_H264_CAPABILITY_PER_PICTURE_TYPE_MIN_MAX_QP_BIT_KHR;
      ext->maxLevelIdc = STD_VIDEO_H264_LEVEL_IDC_6_2;
      ext->maxSliceCount = 1;
      ext->maxPPictureL0ReferenceCount = 1;
      ext->maxBPictureL0ReferenceCount = 0;
      ext->maxL1ReferenceCount = 0;
      ext->maxTemporalLayerCount = 1;
      ext->expectDyadicTemporalLayerPattern = false;
      ext->minQp = 10;
      ext->maxQp = 51;
      ext->prefersGopRemainingFrames = false;
      ext->requiresGopRemainingFrames = false;
      ext->stdSyntaxFlags = VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_UNSET_BIT_KHR;
   }

   strcpy(pCapabilities->stdHeaderVersion.extensionName, VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_EXTENSION_NAME);
   pCapabilities->stdHeaderVersion.specVersion = VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_SPEC_VERSION;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_GetPhysicalDeviceVideoFormatPropertiesKHR(VkPhysicalDevice physicalDevice,
                                               const VkPhysicalDeviceVideoFormatInfoKHR *pVideoFormatInfo,
                                               uint32_t *pVideoFormatPropertyCount,
                                               VkVideoFormatPropertiesKHR *pVideoFormatProperties)
{
   const struct VkVideoProfileListInfoKHR *video_profile_list = vk_find_struct_const(pVideoFormatInfo->pNext, VIDEO_PROFILE_LIST_INFO_KHR);
   for (unsigned p = 0; p < video_profile_list->profileCount; p++) {
      const VkVideoProfileInfoKHR *video_profile = video_profile_list->pProfiles + p;
      if (video_profile->videoCodecOperation != VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR)
         return VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR;

      if (video_profile->chromaSubsampling != VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR ||
            video_profile->lumaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR ||
            video_profile->chromaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)
         return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

      const VkVideoEncodeH264ProfileInfoKHR *h264_profile =
         vk_find_struct_const(video_profile->pNext, VIDEO_ENCODE_H264_PROFILE_INFO_KHR);
      if (h264_profile->stdProfileIdc != STD_VIDEO_H264_PROFILE_IDC_BASELINE)
         return VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR;
   }

   // DPB pictures do not support any other usages
   if (pVideoFormatInfo->imageUsage & VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR && pVideoFormatInfo->imageUsage != VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR)
      return VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR;
   // picture must be either DPB or SRC
   if (!(pVideoFormatInfo->imageUsage & VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR || pVideoFormatInfo->imageUsage & VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR))
      return VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR;


   VK_OUTARRAY_MAKE_TYPED(VkVideoFormatPropertiesKHR, out, pVideoFormatProperties, pVideoFormatPropertyCount);
   const VkFormat supportedFormats[] = {VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM};
   for (int i = 0; i < sizeof(supportedFormats) / sizeof(supportedFormats[0]); ++i) {
      vk_outarray_append_typed(VkVideoFormatPropertiesKHR, &out, p)
      {
         p->format = supportedFormats[i];
         p->componentMapping.r = VK_COMPONENT_SWIZZLE_IDENTITY;
         p->componentMapping.g = VK_COMPONENT_SWIZZLE_IDENTITY;
         p->componentMapping.b = VK_COMPONENT_SWIZZLE_IDENTITY;
         p->componentMapping.a = VK_COMPONENT_SWIZZLE_IDENTITY;
         p->imageCreateFlags = 0;
         if (pVideoFormatInfo->imageUsage & VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR)
            p->imageCreateFlags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
         p->imageType = VK_IMAGE_TYPE_2D;
         p->imageTiling = VK_IMAGE_TILING_OPTIMAL;
         p->imageUsageFlags = pVideoFormatInfo->imageUsage;
      }
      if (pVideoFormatInfo->imageUsage & VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR)
         break;
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_GetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR(VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR* pQualityLevelInfo,
   VkVideoEncodeQualityLevelPropertiesKHR* pQualityLevelProperties)
{
   if (pQualityLevelInfo->pVideoProfile->videoCodecOperation != VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR)
      return VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR;

   if (pQualityLevelInfo->pVideoProfile->chromaSubsampling != VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR ||
         pQualityLevelInfo->pVideoProfile->lumaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR ||
         pQualityLevelInfo->pVideoProfile->chromaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)
      return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

   const VkVideoEncodeH264ProfileInfoKHR *h264_profile =
      vk_find_struct_const(pQualityLevelInfo->pVideoProfile->pNext, VIDEO_ENCODE_H264_PROFILE_INFO_KHR);
   if (h264_profile->stdProfileIdc != STD_VIDEO_H264_PROFILE_IDC_BASELINE)
      return VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR;

   pQualityLevelProperties->preferredRateControlLayerCount = 0;
   pQualityLevelProperties->preferredRateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;

   VkVideoEncodeH264QualityLevelPropertiesKHR *h264_properties =
      vk_find_struct(pQualityLevelProperties->pNext, VIDEO_ENCODE_H264_QUALITY_LEVEL_PROPERTIES_KHR);
   h264_properties->preferredRateControlFlags = VK_VIDEO_ENCODE_H264_RATE_CONTROL_REGULAR_GOP_BIT_KHR | VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR;
   h264_properties->preferredGopFrameCount = 30;
   h264_properties->preferredIdrPeriod = 30;
   h264_properties->preferredConsecutiveBFrameCount = 0;
   h264_properties->preferredTemporalLayerCount = 0;
   h264_properties->preferredConstantQp.qpI = 25;
   h264_properties->preferredConstantQp.qpP = 28;
   h264_properties->preferredConstantQp.qpB = 31;
   h264_properties->preferredMaxL0ReferenceCount = 1;
   h264_properties->preferredMaxL1ReferenceCount = 0;
   h264_properties->preferredStdEntropyCodingModeFlag = VK_FALSE;

   return VK_SUCCESS;
}

VkResult
lvp_CreateVideoSessionKHR(VkDevice _device,
                           const VkVideoSessionCreateInfoKHR *pCreateInfo,
                           const VkAllocationCallbacks *pAllocator,
                           VkVideoSessionKHR *pVideoSession)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);

   struct lvp_video_session *vid =
      vk_alloc2(&device->vk.alloc, pAllocator, sizeof(*vid), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!vid)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   memset(vid, 0, sizeof(struct lvp_video_session));

   VkResult result = vk_video_session_init(&device->vk,
                                           &vid->vk,
                                           pCreateInfo);
   if (result != VK_SUCCESS) {
      vk_free2(&device->vk.alloc, pAllocator, vid);
      return result;
   }

   H264E_create_param_t create_param;
   create_param.width = pCreateInfo->maxCodedExtent.width;
   create_param.height = pCreateInfo->maxCodedExtent.height;
   int sizeof_persist = 0, sizeof_scratch = 0, error;

   error = H264E_sizeof(&create_param, &sizeof_persist, &sizeof_scratch);
   if (error)
   {
      printf("H264E_sizeof error = %d\n", error);
      vk_free2(&device->vk.alloc, pAllocator, vid);
      return VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR;
   }
   printf("sizeof_persist = %d sizeof_scratch = %d\n", sizeof_persist, sizeof_scratch);
   if (pCreateInfo->pictureFormat == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM)
   {
      // allocate memory for splitting UV planes
      int sizeof_split_buffers = (pCreateInfo->maxCodedExtent.width / 2 * pCreateInfo->maxCodedExtent.height / 2) * 2;
      printf("sizeof_split_buffers = %d\n", sizeof_split_buffers);
      vid->split_buffers = vk_alloc2(&device->vk.alloc, pAllocator, sizeof_split_buffers, 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!vid->split_buffers)
      {
         vk_free2(&device->vk.alloc, pAllocator, vid);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
   }
   else {
      vid->split_buffers = NULL;
   }
   vid->enc = vk_alloc2(&device->vk.alloc, pAllocator, sizeof_persist, 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   vid->scratch = vk_alloc2(&device->vk.alloc, pAllocator, sizeof_scratch, 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!vid->enc || !vid->scratch)
   {
      vk_free2(&device->vk.alloc, pAllocator, vid->split_buffers);
      vk_free2(&device->vk.alloc, pAllocator, vid->enc);
      vk_free2(&device->vk.alloc, pAllocator, vid->scratch);
      vk_free2(&device->vk.alloc, pAllocator, vid);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   error = H264E_init(vid->enc, &create_param);
   if (error)
   {
      printf("H264E_init error = %d\n", error);
      vk_free2(&device->vk.alloc, pAllocator, vid->split_buffers);
      vk_free2(&device->vk.alloc, pAllocator, vid->enc);
      vk_free2(&device->vk.alloc, pAllocator, vid->scratch);
      vk_free2(&device->vk.alloc, pAllocator, vid);
      return VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR;
   }

   *pVideoSession = lvp_video_session_to_handle(vid);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
lvp_DestroyVideoSessionKHR(VkDevice _device, VkVideoSessionKHR _session, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(lvp_device, device, _device);
   VK_FROM_HANDLE(lvp_video_session, vid, _session);
   if (!_session)
      return;

   vk_free2(&device->vk.alloc, pAllocator, vid->split_buffers);
   vk_free2(&device->vk.alloc, pAllocator, vid->enc);
   vk_free2(&device->vk.alloc, pAllocator, vid->scratch);

   vk_object_base_finish(&vid->vk.base);
   vk_free2(&device->vk.alloc, pAllocator, vid);
}

VkResult
lvp_GetVideoSessionMemoryRequirementsKHR(VkDevice _device,
                                          VkVideoSessionKHR videoSession,
                                          uint32_t *pMemoryRequirementsCount,
                                          VkVideoSessionMemoryRequirementsKHR *pMemoryRequirements)
{
   //LVP_FROM_HANDLE(lvp_device, device, _device);
   //LVP_FROM_HANDLE(lvp_video_session, vid, videoSession);

   VK_OUTARRAY_MAKE_TYPED(VkVideoSessionMemoryRequirementsKHR, out, pMemoryRequirements, pMemoryRequirementsCount);
   
   /* no memory requirements */

   return vk_outarray_status(&out);
}

VkResult
lvp_BindVideoSessionMemoryKHR(VkDevice _device,
                              VkVideoSessionKHR videoSession,
                              uint32_t bind_mem_count,
                              const VkBindVideoSessionMemoryInfoKHR *bind_mem)
{
   return VK_SUCCESS;
}

VkResult
lvp_CreateVideoSessionParametersKHR(VkDevice _device,
                                     const VkVideoSessionParametersCreateInfoKHR *pCreateInfo,
                                     const VkAllocationCallbacks *pAllocator,
                                     VkVideoSessionParametersKHR *pVideoSessionParameters)
{
   LVP_FROM_HANDLE(lvp_device, device, _device);
   LVP_FROM_HANDLE(lvp_video_session, vid, pCreateInfo->videoSession);
   LVP_FROM_HANDLE(lvp_video_session_params, templ, pCreateInfo->videoSessionParametersTemplate);
   struct lvp_video_session_params *params =
      vk_alloc2(&device->vk.alloc, pAllocator, sizeof(*params), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!params)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result = vk_video_session_parameters_init(&device->vk,
                                                      &params->vk,
                                                      &vid->vk,
                                                      templ ? &templ->vk : NULL,
                                                      pCreateInfo);
   if (result != VK_SUCCESS) {
      vk_free2(&device->vk.alloc, pAllocator, params);
      return result;
   }

   params->has_sps_overides = false;
   params->has_pps_overides = false;
   for (uint32_t i = 0; i < params->vk.h264_enc.h264_sps_count; ++i) {
      StdVideoH264SequenceParameterSet *sps = &params->vk.h264_enc.h264_sps[i].base;
      // TODO: replace example override with correct SPS overrides
      if (!sps->flags.constraint_set1_flag) {
         sps->flags.constraint_set1_flag = 1u;
         params->has_sps_overides = true;
      }
   }

   for (uint32_t i = 0; i < params->vk.h264_enc.h264_pps_count; ++i) {
      StdVideoH264PictureParameterSet *pps = &params->vk.h264_enc.h264_pps[i].base;
      // TODO: replace example override with correct PPS overrides
      pps;
      //params->has_pps_overides = true;
   }

   *pVideoSessionParameters = lvp_video_session_params_to_handle(params);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
lvp_DestroyVideoSessionParametersKHR(VkDevice _device, VkVideoSessionParametersKHR _params,
                                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(lvp_device, device, _device);
   VK_FROM_HANDLE(lvp_video_session_params, params, _params);
   if (!_params)
      return;

   vk_video_session_parameters_finish(&device->vk, &params->vk);
   vk_free2(&device->vk.alloc, pAllocator, params);
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_UpdateVideoSessionParametersKHR(VkDevice _device, VkVideoSessionParametersKHR _params,
                                      const VkVideoSessionParametersUpdateInfoKHR* pUpdateInfo)
{
   // TODO: implement
   assert(0);
   return VK_ERROR_OUT_OF_HOST_MEMORY;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_GetEncodedVideoSessionParametersKHR(VkDevice device,
                                         const VkVideoEncodeSessionParametersGetInfoKHR *pVideoSessionParametersInfo,
                                         VkVideoEncodeSessionParametersFeedbackInfoKHR *pFeedbackInfo,
                                         size_t *pDataSize, void *pData)
{
   VK_FROM_HANDLE(lvp_video_session_params, templ, pVideoSessionParametersInfo->videoSessionParameters);
   size_t total_size = 0;
   size_t size_limit = 0;

   if (pData)
      size_limit = *pDataSize;

   switch (templ->vk.op) {
   case VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR: {
      const struct VkVideoEncodeH264SessionParametersGetInfoKHR *h264_get_info =
         vk_find_struct_const(pVideoSessionParametersInfo->pNext, VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_KHR);
      if (h264_get_info->writeStdSPS) {
         const StdVideoH264SequenceParameterSet *sps =
            vk_video_find_h264_enc_std_sps(&templ->vk, h264_get_info->stdSPSId);
         assert(sps);
         vk_video_encode_h264_sps(sps, size_limit, &total_size, pData);
      }
      if (h264_get_info->writeStdPPS) {
         const StdVideoH264PictureParameterSet *pps =
            vk_video_find_h264_enc_std_pps(&templ->vk, h264_get_info->stdPPSId);
         assert(pps);
         vk_video_encode_h264_pps(pps, templ->vk.h264_enc.profile_idc == STD_VIDEO_H264_PROFILE_IDC_HIGH, size_limit,
                                  &total_size, pData);
      }
      break;
   }
   case VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR: {
      const struct VkVideoEncodeH265SessionParametersGetInfoKHR *h265_get_info =
         vk_find_struct_const(pVideoSessionParametersInfo->pNext, VIDEO_ENCODE_H265_SESSION_PARAMETERS_GET_INFO_KHR);
      if (h265_get_info->writeStdVPS) {
         const StdVideoH265VideoParameterSet *vps = vk_video_find_h265_enc_std_vps(&templ->vk, h265_get_info->stdVPSId);
         assert(vps);
         vk_video_encode_h265_vps(vps, size_limit, &total_size, pData);
      }
      if (h265_get_info->writeStdSPS) {
         const StdVideoH265SequenceParameterSet *sps =
            vk_video_find_h265_enc_std_sps(&templ->vk, h265_get_info->stdSPSId);
         assert(sps);
         vk_video_encode_h265_sps(sps, size_limit, &total_size, pData);
      }
      if (h265_get_info->writeStdPPS) {
         const StdVideoH265PictureParameterSet *pps =
            vk_video_find_h265_enc_std_pps(&templ->vk, h265_get_info->stdPPSId);
         assert(pps);
         vk_video_encode_h265_pps(pps, size_limit, &total_size, pData);
      }
      break;
   }
   default:
      break;
   }

   if (pFeedbackInfo) {
      pFeedbackInfo->hasOverrides = templ->has_sps_overides || templ->has_pps_overides;

      struct VkVideoEncodeH264SessionParametersFeedbackInfoKHR  *h264_feedback_info =
         vk_find_struct(pFeedbackInfo->pNext, VIDEO_ENCODE_H264_SESSION_PARAMETERS_FEEDBACK_INFO_KHR);
      if (h264_feedback_info) {
         h264_feedback_info->hasStdSPSOverrides = templ->has_sps_overides;
         h264_feedback_info->hasStdPPSOverrides = templ->has_pps_overides;
      }
   }

   *pDataSize = total_size;
   return VK_SUCCESS;
}
