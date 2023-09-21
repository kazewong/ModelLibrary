import jax
from jax._src.distributed import initialize
from kazeML.jax.data2vec.data2vec_dataset import Data2VecDataset
from kazeML.jax.data2vec.data2vec_trainer import Data2VecTrainer, Data2VecTrainerParser
import os
import json



if __name__ == "__main__":
    args = Data2VecTrainerParser().parse_args()

    dataset = Data2VecDataset(args.data_path)

    if args.distributed == True:
        initialize()
        if jax.process_index() == 0:
            print("Total number of process: " + str(jax.process_count()))


        n_processes = jax.process_count()
        if jax.process_index() == 0:
            trainer = Data2VecTrainer(dataset, args, logging=True)
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
            with open(args.output_path + "/args.json", "w") as file:
                output_dict = args.as_dict()
                json.dump(output_dict, file, indent=4)
            trainer.train()
        else:
            trainer = Data2VecTrainer(dataset, args, logging=False)
            trainer.train()
    else:
        trainer = Data2VecTrainer(dataset, args, logging=True)
        trainer.train()
