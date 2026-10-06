import os
import time
import platform

if platform.machine() == 'aarch64':  # ARM only
    os.environ['LD_PRELOAD'] = '/lib/aarch64-linux-gnu/libgomp.so.1'

# LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
from argparse import ArgumentParser
from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.xmr.model import XModel


def main():
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-ds_len", type=int, default=10000000)
    parser.add_argument("-train_path", type=str, required=True)
    parser.add_argument("-labels_path", help="Label file for grouped TSV input; omit for PubTator")
    parser.add_argument("-model_config", type=str, default=".models/xmr4el_base_config.json")
    
    args = parser.parse_args()
    
    start = time.time()
    
    xmodel = XModel.load_config(args.model_config)
    if args.labels_path:
        if xmodel.emb_flag != 1:
            parser.error("Grouped TSV input has no [SEP]: it requires emb_flag 1 in the model config")
        train_data = Preprocessor.load_data_labels_from_file(args.train_path, args.labels_path)
        X_train, Y_train = train_data["corpus"], train_data["labels"]
    else:
        train_data = Preprocessor.load_pubtator_file(args.train_path, window=xmodel.context_window,
                                                abbrev=xmodel.abbrev_expansion)
        X_train, Y_train = Preprocessor.organize_pubtator_output(train_data)
    
    del train_data
    
    xmodel.train(X_train[:args.ds_len], Y_train[:args.ds_len])

    # Save the tree
    save_dir = os.path.join(os.getcwd(), "test/test_data/saved_trees")  # Ensure this path is correct and writable
    xmodel.save(save_dir)

    end = time.time()
    print(f"{end - start} secs of running")

# Here is code
if __name__ == "__main__":
    main()
