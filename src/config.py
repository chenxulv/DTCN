import os, sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath("{}/../data/".format(cur_dir))
SOURCE_DIR = os.path.abspath("{}/".format(cur_dir))
CACHE_DIR = os.path.abspath("{}/cache/".format(cur_dir))
RESULT_DIR = os.path.abspath("{}/results".format(cur_dir))
LOG_DIR = os.path.abspath("{}/results".format(cur_dir))

MODEL_CONFIG = {
    "dataset_name": "sgd_s",
    "pre_training": {
        "store_id": "2",
        "warm_up_steps": 4000,
        "init_lr": 1.0,
        "train_epoch": 100,
        "train_batch_size": 16,
        "window_size": 2,
        "utterance_encoder_layer_number": 3,
        "dialogue_encoder_layer_number": 3,
        "utterance_decoder_layer_number": 3,
        "max_dialogue_length": 32,
        "max_utterance_length": 64,
        "d_model": 256,
        "dim_feedforward": 1024,
        "nhead": 4,
        "dropout": 0.1,
        "pad_token_id": 0,
        "activation": "gelu"
    },
    "joint_training": {
        "store_id": "2",
        "loss_rates": {
            "ct": 0.00,
            "kl": 1.0,
            "ae": 1.0,
            "rs": 10.0
        },
        "max_lr": 1e-3,
        "init_lr": 1.0,
        "train_batch_size": 16,
        "train_epoch": 100,
        "discount": 0.4,
        "init_dialog_labels_file_name": "labels_0_dialogue.json",
        "init_utterance_labels_file_name": "labels_0_utterance.json",
        "base_model_file_name": "base_encoder_epoch_100.pth",
        "dialogue_encoder_layer_number": 3,
        "max_dialogue_length": 32,
        "max_utterance_length": 64,
        "utterance_update_interval": 2,
        "dialogue_update_interval": 2,
        "d_model": 256,
        "dim_feedforward": 1024,
        "nhead": 4,
        "dropout": 0.0,
        "pad_token_id": 0,
        "activation": "gelu"
    },
    "clustering": {
        "dataset": {
            "sgd_s": {
                "utterance": 60,
                "dialogue": 29
            },
            "sgd_m": {
                "utterance": 60,
                "dialogue": 59
            }
        },
        "method": {
            "dialogue": "gmm",
            "utterance": "kmeans",
        }
    }
}
