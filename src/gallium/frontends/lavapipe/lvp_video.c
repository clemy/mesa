#include "lvp_private.h"

#include "vk_video/vulkan_video_codecs_common.h"

/* TODO: check whole file for calling conventions VKAPI_ATTR and VKAPI_CALL */

VkResult
lvp_GetPhysicalDeviceVideoCapabilitiesKHR2(VkPhysicalDevice physicalDevice,
                                           const VkVideoProfileInfoKHR *pVideoProfile,
                                           VkVideoCapabilitiesKHR *pCapabilities)
{
   /* TODO: these capabilities are completely wrong (are for decode) */
   //LVP_FROM_HANDLE(lvp_physical_device, physical_device, physicalDevice);

   pCapabilities->minBitstreamBufferOffsetAlignment = 32;
   pCapabilities->minBitstreamBufferSizeAlignment = 32;
   pCapabilities->maxCodedExtent.width = 4096;
   pCapabilities->maxCodedExtent.height = 4096;
   pCapabilities->flags = VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR;

   struct VkVideoDecodeCapabilitiesKHR *dec_caps = (struct VkVideoDecodeCapabilitiesKHR *)
      vk_find_struct(pCapabilities->pNext, VIDEO_DECODE_CAPABILITIES_KHR);
   if (dec_caps)
      dec_caps->flags = VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR;

   /* H264 allows different luma and chroma bit depths */
   if (pVideoProfile->lumaBitDepth != pVideoProfile->chromaBitDepth)
      return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

   if (pVideoProfile->chromaSubsampling != VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR)
      return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

   switch (pVideoProfile->videoCodecOperation) {
   case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR: {
      struct VkVideoDecodeH264CapabilitiesKHR *ext = (struct VkVideoDecodeH264CapabilitiesKHR *)
         vk_find_struct(pCapabilities->pNext, VIDEO_DECODE_H264_CAPABILITIES_KHR);

      if (pVideoProfile->lumaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)
         return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

      pCapabilities->maxDpbSlots = 17;
      pCapabilities->maxActiveReferencePictures = 16;
      pCapabilities->pictureAccessGranularity.width = 64;
      pCapabilities->pictureAccessGranularity.height = 64;
      pCapabilities->minCodedExtent.width = 64;
      pCapabilities->minCodedExtent.height = 64;

      ext->fieldOffsetGranularity.x = 0;
      ext->fieldOffsetGranularity.y = 0;
      ext->maxLevelIdc = STD_VIDEO_H264_LEVEL_IDC_5_1;
      strcpy(pCapabilities->stdHeaderVersion.extensionName, VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_EXTENSION_NAME);
      pCapabilities->stdHeaderVersion.specVersion = VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_SPEC_VERSION;
      break;
   }
   default:
      break;
   }
   return VK_SUCCESS;
}

#define LVP_MACROBLOCK_WIDTH  16
#define LVP_MACROBLOCK_HEIGHT 16
#define LVP_ENC_MAX_RATE_LAYER 4
#define NUM_H2645_REFS        16

VKAPI_ATTR VkResult VKAPI_CALL
lvp_GetPhysicalDeviceVideoCapabilitiesKHR(VkPhysicalDevice physicalDevice, const VkVideoProfileInfoKHR *pVideoProfile,
                                           VkVideoCapabilitiesKHR *pCapabilities)
{
   //VK_FROM_HANDLE(lvp_physical_device, pdev, physicalDevice);

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
      enc_caps->flags = 0;
      enc_caps->rateControlModes = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR |
                                    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR |
                                    VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR;
      enc_caps->maxRateControlLayers = LVP_ENC_MAX_RATE_LAYER;
      enc_caps->maxBitrate = 1000000000;
      enc_caps->maxQualityLevels = 2;
      enc_caps->encodeInputPictureGranularity.width = 1;
      enc_caps->encodeInputPictureGranularity.height = 1;
      enc_caps->supportedEncodeFeedbackFlags = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR |
                                                VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR;
   }
   pCapabilities->minBitstreamBufferOffsetAlignment = 16;
   pCapabilities->minBitstreamBufferSizeAlignment = 16;
   
   switch (pVideoProfile->videoCodecOperation) {
   case VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR: {
      struct VkVideoEncodeH264CapabilitiesKHR *ext = (struct VkVideoEncodeH264CapabilitiesKHR *)vk_find_struct(
         pCapabilities->pNext, VIDEO_ENCODE_H264_CAPABILITIES_KHR);

      const struct VkVideoEncodeH264ProfileInfoKHR *h264_profile =
         vk_find_struct_const(pVideoProfile->pNext, VIDEO_ENCODE_H264_PROFILE_INFO_KHR);

      if (h264_profile->stdProfileIdc != STD_VIDEO_H264_PROFILE_IDC_BASELINE &&
          h264_profile->stdProfileIdc != STD_VIDEO_H264_PROFILE_IDC_MAIN &&
          h264_profile->stdProfileIdc != STD_VIDEO_H264_PROFILE_IDC_HIGH)
         return VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR;

      if (pVideoProfile->lumaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)
         return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

      pCapabilities->maxDpbSlots = NUM_H2645_REFS;
      pCapabilities->maxActiveReferencePictures = NUM_H2645_REFS;
      ext->flags = VK_VIDEO_ENCODE_H264_CAPABILITY_HRD_COMPLIANCE_BIT_KHR |
                   VK_VIDEO_ENCODE_H264_CAPABILITY_PER_PICTURE_TYPE_MIN_MAX_QP_BIT_KHR;
      ext->maxLevelIdc = STD_VIDEO_H264_LEVEL_IDC_6_2;
      ext->maxSliceCount = 1;
      ext->maxPPictureL0ReferenceCount = 1;
      ext->maxBPictureL0ReferenceCount = 0;
      ext->maxL1ReferenceCount = 0;
      ext->maxTemporalLayerCount = 4;
      ext->expectDyadicTemporalLayerPattern = false;
      ext->minQp = 0;
      ext->maxQp = 51;
      ext->prefersGopRemainingFrames = false;
      ext->requiresGopRemainingFrames = false;
      ext->stdSyntaxFlags = VK_VIDEO_ENCODE_H264_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_KHR |
                            VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_UNSET_BIT_KHR |
                            VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_SET_BIT_KHR |
                            VK_VIDEO_ENCODE_H264_STD_WEIGHTED_BIPRED_IDC_EXPLICIT_BIT_KHR;

      strcpy(pCapabilities->stdHeaderVersion.extensionName, VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_EXTENSION_NAME);
      pCapabilities->stdHeaderVersion.specVersion = VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_SPEC_VERSION;
      break;
   }
   default:
      break;
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
lvp_GetPhysicalDeviceVideoFormatPropertiesKHR(VkPhysicalDevice physicalDevice,
                                               const VkPhysicalDeviceVideoFormatInfoKHR *pVideoFormatInfo,
                                               uint32_t *pVideoFormatPropertyCount,
                                               VkVideoFormatPropertiesKHR *pVideoFormatProperties)
{
   /* TODO: check if we require separate allocates for DPB and decode video. */
   if ((pVideoFormatInfo->imageUsage &
        (VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR | VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR)) ==
       (VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR | VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   VK_OUTARRAY_MAKE_TYPED(VkVideoFormatPropertiesKHR, out, pVideoFormatProperties, pVideoFormatPropertyCount);

   bool need_8bit = true;
   bool need_10bit = false;
   const struct VkVideoProfileListInfoKHR *prof_list =
      (struct VkVideoProfileListInfoKHR *)vk_find_struct_const(pVideoFormatInfo->pNext, VIDEO_PROFILE_LIST_INFO_KHR);
   if (prof_list) {
      for (unsigned i = 0; i < prof_list->profileCount; i++) {
         const VkVideoProfileInfoKHR *profile = &prof_list->pProfiles[i];
         if (profile->lumaBitDepth & VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR)
            need_10bit = true;
      }
   }

   if (need_10bit) {
      vk_outarray_append_typed(VkVideoFormatPropertiesKHR, &out, p)
      {
         p->format = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
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

      if (pVideoFormatInfo->imageUsage & (VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR))
         need_8bit = false;
   }

   if (need_8bit) {
      vk_outarray_append_typed(VkVideoFormatPropertiesKHR, &out, p)
      {
         p->format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
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
   }

   return vk_outarray_status(&out);
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

   *pDataSize = total_size;
   return VK_SUCCESS;
}
