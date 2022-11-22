import sys
sys.path.insert(0, "CMPNN")
from train import *
args = parse_train_args()
modify_train_args(args)
logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
mean_auc_score, std_auc_score = cross_validate(args, logger)
print(f'Results: {mean_auc_score:.5f} +/- {std_auc_score:.5f}')