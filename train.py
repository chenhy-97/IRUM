import torch
import os
import torch.nn as nn
import utils
from config import config
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from valid import valid
from utils import confusion_matrix
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from joint import TransformerEncoderLayer


# 钩子函数
def train(config, train_loader, test_loader, test_loader1, fold):
    statistic_val = np.zeros((1, 8))
    statistic_test = np.zeros((1,8))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TransformerEncoderLayer()

    model = model.to(device)

    model.train()



    if config.loss_function == 'CE':
        criterion = nn.CrossEntropyLoss().to(device)

    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    if config.scheduler == 'cosine':
        lr_lambda = lambda epoch: (epoch * (1 - config.warmup_decay) / config.warmup_epochs + config.warmup_decay) \
            if epoch < config.warmup_epochs else \
            (1 - config.min_lr / config.lr) * 0.5 * (math.cos((epoch - config.warmup_epochs) / (
                        config.epochs - config.warmup_epochs) * math.pi) + 1) + config.min_lr / config.lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif config.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=0.9)

    writer = SummaryWriter(comment='_' + config.model_name + '_' + config.writer_comment + '_' + str(fold))

    print("START TRAINING")
    best_acc = 0
    ckpt_path = os.path.join(config.model_path, config.model_name, config.writer_comment)
    model_save_path = os.path.join(ckpt_path, str(fold))
    final_cm = torch.zeros((config.class_num, config.class_num))
    x1 = [0] * config.epochs
    y1 = [0] * config.epochs
    x2 = [0] * config.epochs
    y2 = [0] * config.epochs
    for epoch in range(config.epochs):
        train_acc = 0.0
        cm = torch.zeros((config.class_num, config.class_num))
        epoch_loss = 0
        train_bar = tqdm(train_loader)

        for i, pack in enumerate(train_bar):
            images = pack['imgs'].to(device)
            labels_s = pack['labels'].to(device)
            labels_segs = pack['region'].to(device)
            view1 = images[:,0:3,:,:]
            view2 = images[:, 3:6, :, :]
            output_j1, xd, yd, loss_f = model(view1,view2)


            output = output_j1
            loss_d = utils.Dice_loss(torch.concat([xd, yd], dim=1), labels_segs)
            loss0 = criterion(output, labels_s)
            loss = loss0 + 0.1 * loss_f + 1.5 * loss_d

            pred = output.argmax(dim=1)

            train_acc += torch.eq(pred, labels_s.to(device)).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cm = confusion_matrix(pred.detach(), labels_s.detach(), cm)
            train_bar.desc = "train epoch[{}/{}] loss:{:.5f} loss1:{:.5f}".format(epoch + 1, args.epochs, loss, loss_d)

        lr_scheduler.step()

        if (epoch + 1) % config.log_step == 0:
            # print('[epoch %d]' % (epoch+1))
            with torch.no_grad():
                result, val_cm = valid(config, model,test_loader1, criterion)
            val_loss, val_acc, sen, spe, auc, pre, f1score, m_iou, m_dice = result
            writer.add_scalar('Val/F1score', f1score, global_step=epoch)
            writer.add_scalar('Val/Pre', pre, global_step=epoch)
            writer.add_scalar('Val/Spe', spe, global_step=epoch)
            writer.add_scalar('Val/Sen', sen, global_step=epoch)
            writer.add_scalar('Val/AUC', auc, global_step=epoch)
            writer.add_scalar('Val/Acc', val_acc, global_step=epoch)
            writer.add_scalar('Val/Val_loss', val_loss, global_step=epoch)
            # print('[epoch {}] loss: {:.4f}, acc: {:.4f}, auc: {:.4f}, f1: {:.4f}'.format((epoch + 1), val_loss, val_acc,
            #                                                                              auc, f1score))

            if epoch > 0:
                if val_acc > best_acc:
                    final_cm = val_cm
                    best_acc = val_acc
                    print("=> saved best model")
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    if config.save_model:
                        torch.save(model.state_dict(), os.path.join(model_save_path, 'bestmodel.pth'))
                    with open(os.path.join(model_save_path, 'result.txt'), 'w') as f:
                        f.write('Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1score: %f'
                                % (val_acc, spe, sen, auc, pre, f1score))

                    with torch.no_grad():
                        result1, val_cm1 = valid(config, model, test_loader, criterion)
                    val_loss1, val_acc1, sen1, spe1, auc1, pre1, f1score1, m_iou1, m_dice1 = result1
                    with open(os.path.join(model_save_path, 'result_test.txt'), 'w') as f:
                        f.write('Best Result:\n')
                        f.write('Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1score: %f'
                                % (val_acc1, spe1, sen1, auc1, pre1, f1score1))
                    print('[epoch {}] loss: {:.4f}, acc: {:.4f}, auc: {:.4f}, f1: {:.4f}, iou: {:4f}, dice:{:4f}'.format((epoch + 1), val_loss1, val_acc1,
                                                                                         auc1, f1score1, m_iou1, m_dice1))

                    statistic_val = np.array([val_acc, spe, sen, auc, pre, f1score, m_iou, m_dice])
                    statistic_test = np.array([val_acc1, spe1, sen1, auc1, pre1, f1score1,m_iou1, m_dice1])



        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Train/LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        writer.add_scalar('Train/Acc', cm.diag().sum() / cm.sum(), global_step=epoch)
        writer.add_scalar('Train/Avg_epoch_loss', avg_epoch_loss, global_step=epoch)

        x1[epoch] = train_acc / train_num
        y1[epoch] = val_acc

        x2[epoch] = avg_epoch_loss
        y2[epoch] = val_loss

    # 输出 acc loss 曲线
    plt.figure(1)
    plt.plot(np.arange(1, config.epochs + 1), x1, label='train_accurate')
    plt.plot(np.arange(1, config.epochs + 1), y1, label='val_accurate')
    plt.legend()
    plt.savefig(os.path.join(model_save_path, 'acc.png'))
    plt.show(block=False)
    plt.pause(1)  # 显示1s
    plt.close()

    plt.figure(2)
    plt.plot(np.arange(1, config.epochs + 1), x2, label='train_loss')
    plt.plot(np.arange(1, config.epochs + 1), y2, label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(model_save_path, 'loss.png'))
    plt.show(block=False)
    plt.pause(1)  # 显示1s
    plt.close()

    # 输出混淆矩阵
    cm = np.array(final_cm)
    con_mat_norm = np.around(cm, decimals=3)
    file = open('class_indices.txt', 'r')
    lines = file.readlines()
    file.close()
    labels = [line.strip() for line in lines]
    matrix = con_mat_norm
    num_classes = config.class_num
    plt.imshow(matrix, cmap=plt.cm.Blues)
    # 设置x轴坐标label
    plt.xticks(range(num_classes), labels, rotation=45)
    # 设置y轴坐标label
    plt.yticks(range(num_classes), labels)
    # 显示colorbar
    plt.colorbar()
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Confusion matrix')

    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2
    for x in range(num_classes):
        for y in range(num_classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            # info = int(matrix[y, x]),round(int(matrix[y, x])/int(matrix.sum(axis=0)[x]),2)
            info = int(matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if int(matrix[y, x]) > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'best_cm.png'))
    plt.show(block=False)
    plt.pause(1)  # 显示1s
    plt.close()

    # 输出混淆矩阵
    cm = np.array(val_cm1)
    con_mat_norm = np.around(cm, decimals=3)
    file = open('class_indices.txt', 'r')
    lines = file.readlines()
    file.close()
    labels = [line.strip() for line in lines]
    matrix = con_mat_norm
    num_classes = config.class_num
    plt.imshow(matrix, cmap=plt.cm.Blues)
    # 设置x轴坐标label
    plt.xticks(range(num_classes), labels, rotation=45)
    # 设置y轴坐标label
    plt.yticks(range(num_classes), labels)
    # 显示colorbar
    plt.colorbar()
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Confusion matrix')

    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2
    for x in range(num_classes):
        for y in range(num_classes):
            info = int(matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if int(matrix[y, x]) > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'best_cm_test.png'))
    plt.show(block=False)
    plt.pause(1)  # 显示1s
    plt.close()

    return statistic_val,statistic_test


def seed_torch(seed=1):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    statistic_val = np.zeros((5, 8))
    statistic_test = np.zeros((5, 8))
    seed_torch(42)
    args = config()
    cv = KFold(n_splits=args.fold)
    fold = 0
    train_set = utils.get_dataset(args.data_path,args.label_path, args.csv_path, args.img_size, mode='train')
    test_set = utils.get_dataset(args.data_path,args.label_path, args.csv_path, args.img_size, mode='test')
    # 外部测试集
    test_set1 = utils.get_dataset(args.data_path,args.label_path, args.test_path, args.img_size, mode='test')
    test_loader1 = DataLoader(test_set1, batch_size=1, shuffle=False, num_workers=args.nw)
    print(len(test_loader1))
    print(args)
    argspath = os.path.join(args.model_path, args.model_name, args.writer_comment)
    if not os.path.exists(argspath):
        os.makedirs(argspath)
    with open(os.path.join(argspath, 'model_info.txt'), 'w') as f:
        f.write(str(args))

    for train_idx, test_idx in cv.split(train_set):
        print("\nCross validation fold %d" % fold)
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_num = len(train_sampler)
        train_set_select = copy.deepcopy(train_set)
        train_set_select.info.drop(test_idx, inplace=True)
        args.cls_num = train_set_select.get_cls_num()
        train_loader = DataLoader(train_set_select, batch_size=args.batch_size, shuffle=True, num_workers=args.nw)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, sampler=test_sampler, num_workers=args.nw)
        statistic_val[fold,:], statistic_test[fold,:] = train(args, train_loader, test_loader,test_loader1, fold)
        fold += 1

    np.savetxt("statistic_val.csv", statistic_val, delimiter=",", fmt='%.4f')
    np.savetxt("statistic_test.csv", statistic_test, delimiter=",", fmt='%.4f')