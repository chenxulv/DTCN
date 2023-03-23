### DTCN model training method

## before training the model, you should configue the related params in config.py file
    MODEL_CONFIG = {
        # the dataset_name for training
        "dataset_name": "sgd_s",

        # the parmas used in pre_training stage
        "pre_training": {
            # the id for storing the results
            "store_id": "2",

            # warm up steps for transformer
            "warm_up_steps": 4000,
            "init_lr": 1.0,

            "train_epoch": 100,
            "train_batch_size": 16,

            # the size of window
            "window_size": 2,

            # the layers number
            "utterance_encoder_layer_number": 3,
            "dialogue_encoder_layer_number": 3,
            "utterance_decoder_layer_number": 3,

            "max_dialogue_length": 32,
            "max_utterance_length": 64,

            # embedding size
            "d_model": 256,
            "dim_feedforward": 1024,
            "nhead": 4,
            "dropout": 0.1,
            "pad_token_id": 0,
            "activation": "gelu"
        },

        # the params used in joint training stage.
        "joint_training": {

            # the store_id for storing results, and for loading the base model trained though pre-training stage.
            "store_id": "2",

            # the loss 
            "loss_rates": {"ct":0.00, "kl": 1.0, "ae": 1.0, "rs": 10.0},

            # the max learn rate
            "max_lr":1e-3,
            "init_lr": 1.0,

            "train_batch_size": 16,
            "train_epoch": 80,

            "discount": 0.4,

            # initial clustering assignments for UCRL and DRL modules.
            "init_dialog_labels_file_name": "labels_0_dialogue.json",
            "init_utterance_labels_file_name": "labels_0_utterance.json",

            # the base model pre-trained though pre-trainining stage.
            "base_model_file_name": "base_encoder_epoch_100.pth",

            # DRL layers
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
                "sgd_s": {"utterance": 60, "dialogue": 29},
                "sgd_m": {"utterance": 60, "dialogue": 59}
            },
            "method": {
                "dialogue": "gmm",
                # for speed up utterance clustering, you can use kmeans to cluster utterances, which has little effect.
                "utterance": "kmeans",
            }
        }
    }

## cd src/train/ first

## first step: run pre_training_model.py
eg. python run pre_training_model.py --gpu 0
or  python run pre_training_model.py --gpu 0 > results.txt

## second step: run run_init_clustering.py for initial dialogue and utterance clustering 
eg. python run run_init_clustering.py --epoch 100


## third step: run joint_training_model.py
eg. python run joint_training_model.py --gpu 0
or  python run joint_training_model.py --gpu 0 > results.txt


## if you want check the clustering results of ACC, Purity, NMI and ARI, run the folling command.
# if the log file is results.txt as the above, run

cat results.txt | grep "acc\|epoch"

# then you can see the following results, the first line is the clustering performance by gmm.

    #epoch 1                                                                                                                                             │
    ==> acc: 0.76713,  purity: 0.80255,  nmi: 0.85583, ari: 0.74775 <==                                                                                  │
    # assignment labels by count matrix==> acc: 0.76713,  purity: 0.80255,  nmi: 0.85583, ari: 0.74775 <==                                               │
    # assignment labels by softmax ==> acc: 0.71363,  purity: 0.74624,  nmi: 0.82331, ari: 0.65361 <==                                                   │
    ==> match pre-iteration acc (utterance): 0.43230                                                                                                     │
    ==> match pre-iteration acc (dialogue) : 0.74930                                                                                                     │
    #epoch 2 

