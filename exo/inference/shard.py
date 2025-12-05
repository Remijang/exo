from dataclasses import dataclass, field

@dataclass(frozen=True)
class TpAttr:
  node_rank: int
  world_size: int

  def to_dict(self) -> dict:
    return {
      "node_rank": self.node_rank,
      "world_size": self.world_size,
    }
  
  def from_dict(data: dict) -> 'TpAttr':
    return TpAttr(**data)

@dataclass(frozen=True)
class Shard:
  model_id: str
  start_layer: int
  end_layer: int
  n_layers: int
  tp_attr: TpAttr

  def __hash__(self):
    return hash((self.model_id, self.start_layer, self.end_layer, self.n_layers,
                 self.tp_attr.node_rank, self.tp_attr.world_size))

  def is_first_layer(self) -> bool:
    return self.start_layer == 0

  def is_last_layer(self) -> bool:
    return self.end_layer == self.n_layers - 1

  def get_layer_count(self) -> int:
    return self.end_layer - self.start_layer + 1

  def to_dict(self) -> dict:
    return {
      "model_id": self.model_id,
      "start_layer": self.start_layer,
      "end_layer": self.end_layer,
      "n_layers": self.n_layers,
      "tp_attr": self.tp_attr.to_dict(),
    }

  def from_dict(data: dict) -> 'Shard':
    return Shard(
      data["model_id"],
      data["start_layer"],
      data["end_layer"],
      data["n_layers"],
      TpAttr.from_dict(data["tp_attr"]) if "tp_attr" in data else TpAttr(0, 1)
    )

  def overlaps(self, other: 'Shard') -> bool:
    return shards_overlap(self, other)


def shards_overlap(shard1: Shard, shard2: Shard) -> bool:
  return (shard1.model_id == shard2.model_id and max(shard1.start_layer, shard2.start_layer) <= min(shard1.end_layer, shard2.end_layer))
