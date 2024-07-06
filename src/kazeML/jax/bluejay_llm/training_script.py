import jax
from jax._src.distributed import initialize
from bluejay_llm.dataloader import ThePileDataset
import numpy as np
import json
import os
from bluejay_llm.trainer import BigParser, BlueJayTrainer


if __name__ == "__main__":
    args = BigParser().parse_args()

    if args.distributed == True:
        initialize()
        if jax.process_index() == 0:
            print("Total number of process: " + str(jax.process_count()))

    data_path = args.data_path
    train_set = ThePileDataset(data_path+'/train.bin', process_id=jax.process_index(), num_processes=jax.process_count())
    test_set = ThePileDataset(data_path+'/valid.bin', process_id=jax.process_index(), num_processes=jax.process_count())

    n_processes = jax.process_count()
    if jax.process_index() == 0:
        trainer = BlueJayTrainer(train_set, test_set, args)
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        with open(args.output_path + "/args.json", "w") as file:
            output_dict = args.as_dict()
            json.dump(output_dict, file, indent=4)
        trainer.train()
    else:
        trainer = BlueJayTrainer(train_set, test_set, args)
        trainer.train()
