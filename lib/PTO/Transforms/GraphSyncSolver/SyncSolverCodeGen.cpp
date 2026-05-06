// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===---------- SyncSolverCodeGen.cpp ---- Graph Sync Solver --------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/SyncSolverCodeGen.h"

#include "PTO/IR/PTO.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

static PipeAttr makePipe(MLIRContext *ctx, PIPE pipe) {
  return PipeAttr::get(ctx, pipe);
}

static EventAttr makeEvent(MLIRContext *ctx, int64_t eventId) {
  return EventAttr::get(ctx, static_cast<EVENT>(eventId));
}

Operation *CodeGenerator::resolveSyncAnchor(OperationBase *opBase) {
  if (!opBase)
    return nullptr;
  if (auto *ph = dyn_cast<PlaceHolder>(opBase)) {
    if (ph->beforeOp)
      return ph->beforeOp->op;
    if (ph->afterOp)
      return ph->afterOp->op;
    if (ph->block)
      return ph->block->getParentOp();
    return nullptr;
  }
  return opBase->op;
}

Location CodeGenerator::resolveSyncLoc(OperationBase *opBase) {
  if (Operation *anchor = resolveSyncAnchor(opBase))
    return anchor->getLoc();
  return funcOp.getLoc();
}

void CodeGenerator::setInsertionPoint(IRRewriter &rewriter,
                                      OperationBase *opBase,
                                      bool insertAfter) {
  Operation *anchor = resolveSyncAnchor(opBase);
  if (!anchor) {
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    return;
  }
  if (insertAfter)
    rewriter.setInsertionPointAfter(anchor);
  else
    rewriter.setInsertionPoint(anchor);
}

void CodeGenerator::emitSyncOp(IRRewriter &rewriter, SyncOp *syncOp) {
  if (auto *barrier = dyn_cast<BarrierOp>(syncOp)) {
    rewriter.create<pto::BarrierOp>(resolveSyncLoc(barrier),
                                    makePipe(rewriter.getContext(),
                                             barrier->pipe));
    return;
  }

  auto *setWait = dyn_cast<SetWaitOp>(syncOp);
  if (!setWait || setWait->eventIds.empty())
    return;

  int64_t eventId = setWait->eventIds.front();
  auto srcAttr = makePipe(rewriter.getContext(), setWait->pipeSrc);
  auto dstAttr = makePipe(rewriter.getContext(), setWait->pipeDst);
  auto eventAttr = makeEvent(rewriter.getContext(), eventId);
  Location loc = resolveSyncLoc(setWait);

  if (isa<SetFlagOp>(setWait)) {
    rewriter.create<pto::SetFlagOp>(loc, srcAttr, dstAttr, eventAttr);
  } else if (isa<WaitFlagOp>(setWait)) {
    rewriter.create<pto::WaitFlagOp>(loc, srcAttr, dstAttr, eventAttr);
  }
}

void CodeGenerator::emitSyncMap(IRRewriter &rewriter, SyncMap &syncMap,
                                bool insertAfter) {
  for (auto &[opBase, syncOps] : syncMap) {
    setInsertionPoint(rewriter, opBase, insertAfter);
    for (auto &syncOp : syncOps)
      emitSyncOp(rewriter, syncOp.get());
  }
}

void CodeGenerator::generateResultOps() {
  IRRewriter rewriter(funcOp.getContext());
  emitSyncMap(rewriter, syncMapBefore, /*insertAfter=*/false);
  emitSyncMap(rewriter, syncMapAfter, /*insertAfter=*/true);
}
