import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/img/')
    parser.add_argument('--label_path', type=str, default='../data/mask/')
    parser.add_argument('--csv_path', type=str, default='../data/train_data.csv')
    parser.add_argument('--test_path', type=str, default='../data/test_data.csv')
    parser.add_argument('--model_name', type=str, default='swin')
    parser.add_argument('--model_path', type=str, default='./weight')
    parser.add_argument('--writer_comment', type=str, default='YN')
    parser.add_argument('--save_model', type=bool, default=True)

    # MODEL PARAMETER
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--nw', type=int, default=8)


    parser.add_argument('--loss_function', type=str, default='CE')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='step', choices=['cosine', 'step'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_decay', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--step', type=int, default=10)

    config = parser.parse_args()
    return config