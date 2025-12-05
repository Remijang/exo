from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition, TpAttr


class PrefillDecodePartitioningStrategy(PartitioningStrategy):
  def partition(self, topology: Topology) -> List[Partition]:
    nodes = list(topology.all_nodes())
    """ 
    sort by fp16 flops (descending)
    first node -> prefill mode
    the others ->  decode mode
    """
    nodes.sort(key=lambda x: (x[1].flops.fp16, x[0]), reverse=True)
    total_memory = sum(node[1].memory for node in nodes)
    partitions = []
    start = 0
    num_decode_nodes = len(nodes) - 1
    for i, node in enumerate(nodes):
      end = round(start + (node[1].memory/total_memory), 5)

      # first node
      if i == 0:
        partitions.append(Partition(node[0], start, end, TpAttr(0, 1)))
      # the rest
      else:
        partitions.append(Partition(node[0], start, end, TpAttr(i - 1, num_decode_nodes)))
      
      start = end
    return partitions
