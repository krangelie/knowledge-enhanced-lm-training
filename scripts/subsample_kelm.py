import json
import math
import os

from tqdm import tqdm
import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import numpy as np
from sklearn.model_selection import train_test_split


def extract_from_idxs(data_list, index_list, out_dir, sample_ratio):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"subsampled_{sample_ratio}.jsonl"), "w") as f:
        for idx in tqdm(index_list):
            sample = json.loads(data_list[idx])
            sample["original_idx"] = str(idx)
            f.write(json.dumps(sample) + "\n")


def subsample_kelm(kelm_json_list, out_dir, sample_ratio=0.01):
    count = len(kelm_json_list)
    n_samples = math.ceil(count * sample_ratio)
    np.random.seed(42)
    sample_idxs = np.random.choice(count, n_samples)
    print("Extracting subsample")
    extract_from_idxs(kelm_json_list, sample_idxs, out_dir, sample_ratio)

    print(f"Final subset contains {n_samples} out of {count} instances")


def create_dev_holdout_splits(cfg, json_list):
    # Split indices randomly into dev and test
    train_idxs, test_idxs = train_test_split(range(len(json_list)), test_size=cfg.test_size,
                                             random_state=cfg.random_state)
    # Store indices for reproducibility
    with open(os.path.join(os.path.dirname(cfg.kelm_path), "holdout_split",
                           f"kelm_holdout_indices_ratio{cfg.test_size}_seed{cfg.random_state}.txt"), 'w') as f:
        for i in test_idxs:
            f.write(str(i))
            f.write("\n")
    # Extract instances via the sampled index lists
    print("Extracting test split")
    extract_from_idxs(json_list, test_idxs,
                      os.path.join(os.path.dirname(cfg.kelm_path), "holdout_split"),
                      cfg.test_size)
    print("Extracting dev split")
    extract_from_idxs(json_list, train_idxs,
                      os.path.join(os.path.dirname(cfg.kelm_path), "dev_split"),
                      1 - cfg.test_size)


@dataclass
class MyConfig:
    kelm_path: str = "./data/kelm/dev_split/subsampled_0.9999.jsonl"
    output_dir: str = "./data/kelm/subsampled"  # used if not create_holdout
    subsample_size: float = 0.50  # for subsampling from full or dev
    create_holdout: bool = False
    test_size: float = 0.0001  # for holdout set creation only
    random_state: int = 42


cs = ConfigStore.instance()
cs.store(name="conf", node=MyConfig())


@hydra.main(version_base=None, config_name="conf")
def main(cfg: MyConfig) -> None:
    with open(cfg.kelm_path) as json_file:
        json_list = list(json_file)

    if cfg.create_holdout:
        create_dev_holdout_splits(cfg, json_list)
    else:
        subsample_kelm(json_list, cfg.output_dir, sample_ratio=cfg.subsample_size)


if __name__ == '__main__':
    main()
