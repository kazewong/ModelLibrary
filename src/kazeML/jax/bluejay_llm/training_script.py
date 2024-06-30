import jax
from jax._src.distributed import initialize
from kazeML.jax.bluejay_llm.bluejay_llm.dataloader import ThePileDataset
import numpy as np
import json
import os
from kazeML.jax.bluejay_llm.bluejay_llm.trainer import BigParser, BlueJayTrainer


if __name__ == "__main__":
    args = BigParser().parse_args()

    if args.distributed == True:
        initialize()
        if jax.process_index() == 0:
            print("Total number of process: " + str(jax.process_count()))

    dataset = ThePileDataset(args.data_path)

    n_processes = jax.process_count()
    if jax.process_index() == 0:
        trainer = BlueJayTrainer(dataset, args)
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        with open(args.output_path + "/args.json", "w") as file:
            output_dict = args.as_dict()
            json.dump(output_dict, file, indent=4)
        trainer.train()
    else:
        trainer = BlueJayTrainer(dataset, args)
        trainer.train()
