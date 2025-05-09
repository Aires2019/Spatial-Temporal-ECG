import argparse


def args():
    """Get default arguments."""
    parser = argparse.ArgumentParser()

    # general configuration
    # parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='PTB_XL', help='choose PTB_XL, CPCS or HFHC')
    parser.add_argument('--save_path', type=str, default='./save')

    # training related
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--early_stop', type=int, default=10, help="Early stopping")

    parser.add_argument('--lr_scheduler', type=bool, default=True)
    parser.add_argument('--show_interval', type=int, default=1)
    parser.add_argument('--evaluation_method', type=str, default='top-k', help='choose top-k or general')

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)

    ## signal related
    parser.add_argument('--patch_size', type=int, default=10)
    parser.add_argument('--time_denpen_len', type=int, default=10)
    parser.add_argument('--num_sensor', type=int, default=14)
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--max_rul', type=int, default=125)
    parser.add_argument('--n_class', type=int, default=5)

    parser.add_argument('--save_name', default='test', type=str)
    parser.add_argument('--gpu', default=2, type=int,
                        help='GPU id to use.')

    # model
    parser.add_argument('--model_name', default='STAR', type=str, help=' choose model to train')
    return parser.parse_args()
