import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    args.config = f"./exps/{args.model_name}.json"
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    # args.update(param)  # Add parameters from json
    param.update(args) 
    train(param)
    #test(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')

    parser.add_argument('--dataset', type=str, default="cifar100")
    parser.add_argument('--memory_size', '-ms', type=int, default=2000)
    parser.add_argument('--init_cls', '-init', type=int, default=25)
    parser.add_argument('--increment', '-incre', type=int, default=5)
    parser.add_argument('--model_name', '-model', type=str, default='ilpc')
    parser.add_argument('--loss_name', type=str, default='PGD_Prototype_Novel')
    parser.add_argument('--convnet_type', '-net', type=str, default='resnet32')
    parser.add_argument('--prefix', '-p', type=str, help='exp type', default='benchmark', choices=['benchmark', 'fair', 'auc'])
    parser.add_argument('--device', '-d', nargs='+', type=int, default=[0])
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--skip', action="store_true")
    
    parser.add_argument('--train_base', action='store_true')
    parser.add_argument('--train_adaptive', action='store_true')

    # init
    parser.add_argument('--scheduler', type=str, default='steplr', choices=['steplr', 'multisteplr', 'cosine'])
    parser.add_argument('--init_epoch', type=int, default=150)
    parser.add_argument('--t_max', type=int, default=None)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--init_milestones', type=list, default=[60, 120, 170])
    parser.add_argument('--init_lr_decay', type=float, default=0.1)
    parser.add_argument('--init_weight_decay', type=float, default=0.0005)
    
    # update
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--milestones', type=list, default=[80, 120, 150])
    parser.add_argument('--lrate_decay', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--weight_decay', type=float, default=2e-4)

    parser.add_argument('--alpha_aux', type=float, default=1.0)

    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--model', default='PointNet', help='model name [default: pointnet]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser


if __name__ == '__main__':
    main()
