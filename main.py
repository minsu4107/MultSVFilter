import torch
import argparse
from torch.utils.data import DataLoader
from src import train
from torch.utils.data import random_split, ConcatDataset

from src.dataset import Multimodal_Datasets

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# SV
parser.add_argument('--caller', type=str, default='lumpy,manta,delly',
                    help='sv_caller_list e.g lumpy,manta,delly')
parser.add_argument('--save_path', type=str, default='./pretrained/',
                    help='save_model_path')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')

parser.add_argument('--train_path', type=str, default='./test_data/', help='path for learning dataset')
parser.add_argument('--train_bed_path', type=str, default='./test_data/', help='path for learning dataset')
parser.add_argument('--test_path', type=str, required = True, help='path for test dataset')
parser.add_argument('--test_bed_path', type=str, required = True, help='path for test dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.4,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.2,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.2,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.4,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.2,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.2,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=1,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_true',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=368, metavar='N',
                    help='batch size (default: 120)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-5)')
parser.add_argument('--optim', type=str, default='RAdam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs (default: 20)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=5,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')

parser.add_argument('--test', action='store_true', default=True, help="Just testing")


args = parser.parse_args()

torch.manual_seed(args.seed)
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        use_cuda = False
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

if args.test == True:
    args.batch_size = 1

def get_data(args):
    print(f"  - Creating new {indi} data")
    data = Multimodal_Datasets(args.data_path)
    
    return data

print("Start loading the data....")

### normal learning
if args.test == False:
    sampling_rate = 0.1
    Train_dataset = Multimodal_Datasets(args.train_path, args.train_bed_path,args.no_cuda)
    Test_dataset = Multimodal_Datasets(args.test_path,args.test_bed_path,args.no_cuda)

    train_size = len(Train_dataset)
    valid_size = int(train_size * 0.1)
    train_size -= valid_size

    train_size = dataset_size - valid_size
    train_data, valid_data = random_split(train_data, [train_size, valid_size], generator=torch.Generator(device='cuda').manual_seed(args.seed))

    if use_cuda == True:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))
else:
    test_data = Multimodal_Datasets(args.test_path, args.test_bed_path, args.no_cuda)
    train_data = test_data
    valid_data = test_data
    
    if use_cuda == True:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cpu'))
print('Finish loading the data....')


####################################################################
# 
# Hyperparameters
# 
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = test_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = test_data.get_seq_len()
hyp_params.layers = 5
hyp_params.use_cuda = use_cuda
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
if args.test == True:
    hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
else:
    hyp_params.n_test = len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = 1
hyp_params.criterion = 'BCELoss'

if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)