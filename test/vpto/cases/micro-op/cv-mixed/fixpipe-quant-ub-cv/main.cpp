// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      const char *_recent = aclGetRecentErrMsg();                                \
      if (_recent != nullptr && _recent[0] != '\0')                              \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);             \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

#define FILE_CHECK(expr, path)                                                   \
  do {                                                                           \
    if (!(expr)) {                                                               \
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\n",        \
                   path, __FILE__, __LINE__);                                    \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchFixpipe_quant_ub_cv_kernel(__fp16 *a, __fp16 *b, float *fp,
                                      __fp16 *out, void *stream);

int main() {
  constexpr size_t kHalfElems = 16 * 16;
  constexpr size_t kFpElems = 32;
  constexpr size_t kSizeA = kHalfElems * sizeof(__fp16);
  constexpr size_t kSizeB = kHalfElems * sizeof(__fp16);
  constexpr size_t kSizeFp = kFpElems * sizeof(float);
  constexpr size_t kSizeOut = kHalfElems * sizeof(__fp16);

  __fp16 *aHost = nullptr;
  __fp16 *bHost = nullptr;
  float *fpHost = nullptr;
  __fp16 *outHost = nullptr;
  __fp16 *aDevice = nullptr;
  __fp16 *bDevice = nullptr;
  float *fpDevice = nullptr;
  __fp16 *outDevice = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t inputSize = 0;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)&aHost, kSizeA));
  ACL_CHECK(aclrtMallocHost((void **)&bHost, kSizeB));
  ACL_CHECK(aclrtMallocHost((void **)&fpHost, kSizeFp));
  ACL_CHECK(aclrtMallocHost((void **)&outHost, kSizeOut));
  ACL_CHECK(aclrtMalloc((void **)&aDevice, kSizeA, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&bDevice, kSizeB, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&fpDevice, kSizeFp, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = kSizeA;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, aHost, kSizeA) && inputSize == kSizeA,
             "./v1.bin");
  inputSize = kSizeB;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, bHost, kSizeB) && inputSize == kSizeB,
             "./v2.bin");
  inputSize = kSizeFp;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, fpHost, kSizeFp) && inputSize == kSizeFp,
             "./v3.bin");
  inputSize = kSizeOut;
  FILE_CHECK(ReadFile("./v4.bin", inputSize, outHost, kSizeOut) &&
                 inputSize == kSizeOut,
             "./v4.bin");

  ACL_CHECK(aclrtMemcpy(aDevice, kSizeA, aHost, kSizeA, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(bDevice, kSizeB, bHost, kSizeB, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(fpDevice, kSizeFp, fpHost, kSizeFp, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outDevice, kSizeOut, outHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchFixpipe_quant_ub_cv_kernel(aDevice, bDevice, fpDevice, outDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outHost, kSizeOut, outDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  FILE_CHECK(WriteFile("./v4.bin", outHost, kSizeOut), "./v4.bin");

cleanup:
  aclrtFree(aDevice);
  aclrtFree(bDevice);
  aclrtFree(fpDevice);
  aclrtFree(outDevice);
  aclrtFreeHost(aHost);
  aclrtFreeHost(bHost);
  aclrtFreeHost(fpHost);
  aclrtFreeHost(outHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
