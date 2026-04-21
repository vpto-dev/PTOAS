// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_HIVMINTRINSICNAMING_H
#define MLIR_DIALECT_PTO_TRANSFORMS_HIVMINTRINSICNAMING_H

#include <string>
#include <vector>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::pto {

struct NamingInputs {
  std::string sourceOpName;
  std::string family;
  std::string vectorShape;
  std::string elementType;
  std::vector<std::string> usedFields;
  std::vector<std::string> missingFields;
};

struct UnresolvedEmissionRecord {
  std::string sourceOpName;
  std::string placeholderName;
  std::string candidateName;
  std::vector<std::string> usedFields;
  std::vector<std::string> missingFields;
  std::string resultTypeFragment;
  std::string location;
};

struct IntrinsicSelection {
  bool resolved = false;
  std::string sourceOpName;
  std::string calleeName;
  std::string placeholderName;
  std::string candidateName;
  std::vector<std::string> usedFields;
  std::vector<std::string> missingFields;
  std::string resultTypeFragment;
  std::string location;

  std::string getEmittedCallee() const {
    return resolved ? calleeName : placeholderName;
  }

  UnresolvedEmissionRecord asUnresolvedRecord() const {
    return UnresolvedEmissionRecord{sourceOpName, placeholderName, candidateName,
                                    usedFields, missingFields, resultTypeFragment,
                                    location};
  }
};

FailureOr<IntrinsicSelection> selectIntrinsic(Operation *op);
FailureOr<IntrinsicSelection> selectLoadIntrinsic(Operation *op);
FailureOr<IntrinsicSelection> selectUnaryIntrinsic(Operation *op);
FailureOr<IntrinsicSelection> selectStoreIntrinsic(Operation *op);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_HIVMINTRINSICNAMING_H
