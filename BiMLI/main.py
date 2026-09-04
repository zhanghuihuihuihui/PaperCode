import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)
import time
import wandb
from args import get_args,get_all_logger
from models_trains.train_mutil_gpu import train_mutil_gpu
from models_trains.train_no_mutil_gpu import train_no_mutil_gpu
from models.models import BiMLI
from args import get_hits,get_data
import os

args = get_args()
hits = get_hits()
tri_datas,emb_data = get_data(args)
# all_triplets,train_triplets,test_triplets,val_triplets
# config = args
loggerss = get_all_logger(args)
model_name = {
    # Ablation experiment
    'BiMLI': BiMLI
}
def train():
    device = args.device
    if args.every_loss_verbose == 2:
 
       params = f'dataset-{args.dataset}-seed-{args.seed}-lm-{args.model_lm_embd}-ml-{args.model_ml_embd}-agg-{args.model_mlm_agg}-cl-{args.mm_select}'
       wandb.init(project=args.model_select_wandb, name = params)
       wandb.config.update(args)
       args_config = wandb.config
    
    train_no_mutil_gpu(args_config,model_name,loggerss,hits,tri_datas,emb_data)



if __name__ == "__main__":
    # train()
    