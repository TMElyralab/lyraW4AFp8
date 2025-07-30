from typing import Any, Dict, List, Tuple, TypedDict, Optional

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    CLUSTER_SIZE: int
    SCHE: str


def get_configs_compute_bound() -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []

    for block_m in [128, 256]:
        for block_n in [16, 32, 64, 128, 256]:
            for cluster in [1, 2]:
                for SCHE in ["streamK", "PersistentScheduler"]:
                    configs.append(
                        {"BLOCK_SIZE_M": block_m,
                        "BLOCK_SIZE_N": block_n,
                        "CLUSTER_SIZE": cluster,
                        "SCHE": SCHE
                        }
                    )
    return configs


def get_schedule_name(config):
    schedule = "{}x{}_{}x1x1_TmaMI__TmaCoop_{}".format(
        config["BLOCK_SIZE_M"], 
        config["BLOCK_SIZE_N"],
        config["CLUSTER_SIZE"],
        config["SCHE"])
    return schedule


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "CLUSTER_SIZE": config["CLUSTER_SIZE"],
        "SCHE": config["SCHE"],
    }