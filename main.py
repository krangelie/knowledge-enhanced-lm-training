import os

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore

from train import train

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class GPT2Config:
    model_name_or_path: str = "gpt2"  # GPT-2 small
    model_dim: int = 768
    context_len: int = 1024
    #dropout: float = 0.2
    #learning_rate: float = 1e-5
    #adam_epsilon: float = 1e-8
    train_batch_size: int = 32
    eval_batch_size: int = 32
    epochs: int = 3
    max_seq_length: int = 512
    #use_weights: bool = True


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
    model: GPT2Config = GPT2Config()
    data: DataConfig = KelmQuarter()
    output_dir: str = "/export/home/kraft/data/kelm/output"


cs = ConfigStore.instance()
cs.store(name="conf", node=MyConfig())


@hydra.main(version_base=None, config_name="conf")
def my_app(cfg: MyConfig) -> None:
    print(f"Finetuning {cfg.model.model_name_or_path} on {cfg.data.dataset_version} version of KELM")
    train(cfg)


if __name__ == '__main__':
    my_app()

