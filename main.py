import os

from torch.cuda import is_available
from dataclasses import dataclass
from omegaconf import MISSING
import hydra
from hydra.core.config_store import ConfigStore

#os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

DEVICE = "cuda" if is_available() else "cpu"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ModelConfig:
    model_name_or_path: str = MISSING
    model_dim: int = MISSING
    dropout: float = 0.0
    learning_rate: float = 1e-5
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    weight_decay: float = 0.0
    train_batch_size: int = 32
    eval_batch_size: int = 32
    max_epochs: int = 5
    max_seq_length: int = 512
    devices: int = 1  # num of gpus
    num_workers: int = 1
    use_weights: bool = False


@dataclass
class GPT2Config(ModelConfig):
    model_name_or_path = "gpt2"  # GPT-2 small
    model_dim = 768
    dropout: float = 0.2
    learning_rate: float = 1e-5
    adam_epsilon: float = 1e-8
    train_batch_size: int = 32
    eval_batch_size: int = 32
    max_epochs: int = 5
    max_seq_length: int = 512
    use_weights: bool = True


@dataclass
class DataConfig:
    data_path: str = MISSING


@dataclass
class KelmDebugSize(DataConfig):
    data_path = "."


@dataclass
class KelmFull(DataConfig):
    data_path = "."


@dataclass
class MyConfig:
    model: ModelConfig = GPT2Config()
    data: DataConfig = KelmDebugSize()


cs = ConfigStore.instance()
cs.store(name="conf", node=MyConfig())


@hydra.main(version_base=None, config_name="conf")
def my_app(cfg: MyConfig) -> None:
    #train.main(cfg)
    pass


if __name__ == '__main__':
    my_app()

