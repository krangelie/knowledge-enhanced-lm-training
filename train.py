from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import NeptuneLogger


def train(cfg):
    seed_everything(42)
    neptune_logger = NeptuneLogger(project="kraft-ml/KE-PLM",
                                   #capture_stdout=False,
                                   #capture_stderr=False,
                                   #capture_hardware_metrics=False,
                                   )
    # load data model
    # load model
    # instantiate trainer
    # train
    #    neptune_logger.log_model_summary(model=model, max_depth=-1)
