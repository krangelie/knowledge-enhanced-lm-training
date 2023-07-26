import os

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore

from train import train


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ModelConfig:
    model_name: str = ""
    checkpoint: str = ""
    model_dim: int = 0
    context_len: int = 0
    train_batch_size: int = 0
    eval_batch_size: int = 0
    epochs: int = 0
    lr: float = 0.00
    weight_decay: float = 0.01
    load_best_model_at_end: bool = False
    mlm: bool = False


@dataclass
class GPT2Config(ModelConfig):
    model_name: str = "gpt2-medium"  # "gpt2" (small) d=768, "gpt2-medium" d=1024, "gpt2-large" d=1280
    checkpoint: str = ""
    model_dim: int = 1024
    context_len: int = 1024
    train_batch_size: int = 16
    eval_batch_size: int = 16
    epochs: int = 3
    lr: float = 0.00005


@dataclass
class RobertaConfig(ModelConfig):
    model_name: str = "roberta-base"
    checkpoint: str = ""
    model_dim: int = 768
    context_len: int = 512
    train_batch_size: int = 16
    eval_batch_size: int = 16
    epochs: int = 3
    lr: float = 0.00005
    mlm: bool = True


@dataclass
class DataConfig:
    test_data_path: str = "/export/home/kraft/data/kelm/holdout_split/subsampled_0.0001.jsonl"


@dataclass
class KelmMini(DataConfig):
    dataset_version: str = "mini"
    train_data_path: str = "/export/home/kraft/data/kelm/subsampled/subsampled_0.1.jsonl"


@dataclass
class KelmQuarter(DataConfig):
    dataset_version: str = "quarter"
    train_data_path: str = "/export/home/kraft/data/kelm/subsampled/subsampled_0.25.jsonl"


@dataclass
class KelmHalf(DataConfig):
    dataset_version: str = "half"
    train_data_path: str = "/export/home/kraft/data/kelm/subsampled/subsampled_0.5.jsonl"


@dataclass
class KelmFull(DataConfig):
    dataset_version: str = "full"
    train_data_path: str = "/export/home/kraft/data/kelm/kelm_generated_corpus.jsonl"


@dataclass
class MyConfig:
    model: ModelConfig = GPT2Config()
    data: DataConfig = KelmFull()
    output_dir: str = "/export/home/kraft/data/kelm/output"


cs = ConfigStore.instance()
cs.store(name="conf", node=MyConfig())


@hydra.main(version_base=None, config_name="conf")
def my_app(cfg: MyConfig) -> None:
    print(f"Finetuning {cfg.model.model_name} on {cfg.data.dataset_version} version of KELM")
    train(cfg)


if __name__ == '__main__':
    my_app()

