import jax
from jax._src.distributed import initialize
from kazeML.jax.diffusion.diffusion_dataset import DiffusionDataset
import numpy as np
import json
import os
from kazeML.jax.diffusion.sde_score_trainer import BigParser, SDEDiffusionTrainer


if __name__ == "__main__":
    args = BigParser().parse_args()

    if args.distributed == True:
        initialize()
        if jax.process_index() == 0:
            print("Total number of process: " + str(jax.process_count()))

    dataset = DiffusionDataset(args.data_path)

    n_processes = jax.process_count()
    if jax.process_index() == 0:
        trainer = SDEDiffusionTrainer(dataset, args, logging=True)
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        with open(args.output_path + "/args.json", "w") as file:
            output_dict = args.as_dict()
            output_dict["data_shape"] = trainer.data_shape
            json.dump(output_dict, file, indent=4)
        trainer.train()
    else:
        trainer = SDEDiffusionTrainer(dataset, args, logging=False)
        trainer.train()
