/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "graph/topo.h"

hipError_t strideMemcpyAsync(void *dst, const void *src, const size_t size, const int height, const int width, hipStream_t stream);

NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream) {
  // Determine Pivot A2A support now that we know number of channels
  comm->topo->pivotA2AEnabled = comm->topo->pivotA2AEnabled && comm->nChannels >= comm->topo->pivotA2ANumBiRings * 2;
  if (comm->topo->pivotA2AEnabled) {
    struct ncclInfo info = { ncclFuncAllToAllPivot, "AllToAllPivot",
      sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS };
    return ncclEnqueueCheck(&info);
  } else {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = count * ncclTypeSize(datatype);
    if (count == 0) return ncclSuccess;
    /*
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, datatype, r, comm, stream));
      NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count, datatype, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    */
    int nGpus = comm->localRanks, nNodes = comm->nNodes;
    if (nGpus == 1 || nNodes == 1) {
      WARN("number of local GPUs (%d) or number of nodes (%d) is 1.", nGpus, nNodes);
      return ncclInvalidUsage;
    }
    // 2D Hierarchical AlltoAll algorithm
    // phase 0. per-gpu (nGpus) stride copy
    CUDACHECK(strideMemcpyAsync(recvbuff, sendbuff, rankOffset, nGpus, nNodes, stream));
    // phase 1. intra-node alltoall
    NCCLCHECK(ncclGroupStart());
    for (int g = 0; g < nGpus; g++) {
      NCCLCHECK(ncclSend(((char*)recvbuff) + g * nNodes * rankOffset, nNodes * count, datatype, g + comm->node * nGpus, comm, stream));
      NCCLCHECK(ncclRecv(((char*)sendbuff) + g * nNodes * rankOffset, nNodes * count, datatype, g + comm->node * nGpus, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    // phase 2. per-gpu (nNodes) stride copy
    CUDACHECK(strideMemcpyAsync(recvbuff, sendbuff, rankOffset, nNodes, nGpus, stream));
    // phase 3. inter-node alltoall
    NCCLCHECK(ncclGroupStart());
    for (int n = 0; n < nNodes; n++) {
      NCCLCHECK(ncclSend(((char*)recvbuff) + n * nGpus * rankOffset, nGpus * count, datatype, n * nGpus + comm->cudaDev, comm, stream));
      NCCLCHECK(ncclRecv(((char*)sendbuff) + n * nGpus * rankOffset, nGpus * count, datatype, n * nGpus + comm->cudaDev, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(hipMemcpyAsync(recvbuff, sendbuff, comm->nRanks * rankOffset, hipMemcpyDeviceToDevice, stream));
    return ncclSuccess;
  }
}
