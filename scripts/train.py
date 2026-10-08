import os
import time
import platform
import logging

if platform.machine() == 'aarch64':  # ARM only
    os.environ['LD_PRELOAD'] = '/lib/aarch64-linux-gnu/libgomp.so.1'

# LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
from argparse import ArgumentParser
from xmr4el import set_verbosity
from xmr4el.data.readers import Preprocessor
from xmr4el.xmodel import XModel


def main():
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-ds_len", type=int, default=10000000)
    parser.add_argument("-train_path", type=str, required=True)
    parser.add_argument("-labels_path", help="Label file for grouped TSV input; omit for PubTator")
    parser.add_argument("-model_config", type=str, default="configs/xmr4el_base_config.json")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-verbose", action="store_true", help="Include DEBUG diagnostics")
    verbosity.add_argument("-quiet", action="store_true", help="Show only warnings and errors")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    start = time.perf_counter()
    
    xmodel = XModel.load_config(args.model_config)
    set_verbosity(0 if args.quiet else 2 if args.verbose else 1)
    logger = logging.getLogger("xmr4el.train")
    logger.info("Run started: input=%s labels=%s config=%s ds_len=%d",
                args.train_path, args.labels_path, args.model_config, args.ds_len)
    logger.info("Loading training data")
    if args.labels_path:
        if xmodel.features != "tfidf":
            parser.error('Grouped TSV input has no [SEP]: it requires "features": "tfidf" in the model config')
        train_data = Preprocessor.load_data_labels_from_file(args.train_path, args.labels_path)
        X_train, Y_train = train_data["corpus"], train_data["labels"]
    else:
        train_data = Preprocessor.load_pubtator_file(args.train_path, window=xmodel.context_window,
                                                abbrev=xmodel.abbrev_expansion)
        X_train, Y_train = Preprocessor.organize_pubtator_output(train_data)
    
    del train_data
    logger.info("Data loaded: selected_groups=%d/%d elapsed=%.1fs",
                len(Y_train[:args.ds_len]), len(Y_train), time.perf_counter() - start)
    
    xmodel.train(X_train[:args.ds_len], Y_train[:args.ds_len])

    # Save the tree
    save_dir = os.path.join(os.getcwd(), "outputs/saved_trees")  # Ensure this path is correct and writable
    xmodel.save(save_dir)

    logger.info("Run completed: elapsed=%.1fs", time.perf_counter() - start)

# Here is code
if __name__ == "__main__":
    main()
