import time
import argparse
import logging
from data_loader import *
from models import *
from optimize import *
from utils import *
from datetime import datetime
from earlystopping import EarlyStoppingF1
import gc



def evaluate(model, loader, domain, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for batch in loader:
            adj, feats, labels, vertices = [tmp.to(device) for tmp in batch]
            out = model(feats, vertices, adj, domain, recon=False)
            y_true += labels.data.tolist()
            y_score += out['cls_output'][:, 1].data.tolist()

    auc, f1 = get_metrics(y_true, y_score)
    print('Eval AUC={:.6f} F1={:.6f}'.format(auc, f1))
    return auc, f1

def run(args):
    timestr = datetime.now().strftime("%Y%m%d%H%M%S")
    logger = get_logger(os.path.join('logs', f'DGDA_src_{args.src_data}_{timestr}.log'))
    logger.info(args)
    dataset = ['oag', 'twitter', 'weibo', 'digg']
    dataset.remove(args.src_data)

    if args.src_data == 'twitter':
        dataset.remove('oag')
        dataset.remove('digg')


    src_ds, n_feat, src_class_weight, src_train_loader, src_val_loader, src_test_loader = load_influence_dataset(
        path=args.data_path + args.src_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        seed=args.seed,
        num_workers=args.num_workers)

    device = torch.device('cuda:{}'.format(args.device))
    src_class_weight = torch.FloatTensor(src_class_weight).to(device)
    recons_weight = torch.FloatTensor([args.recons_weight]).to(device)
    beta = torch.FloatTensor([args.beta]).to(device)
    ent_weight = torch.FloatTensor([args.ent_weight]).to(device)
    d_w = torch.FloatTensor([args.d_weight]).to(device)
    y_w = torch.FloatTensor([args.y_weight]).to(device)
    m_w = torch.FloatTensor([args.m_weight]).to(device)

    weights = [src_class_weight, recons_weight, beta, ent_weight, d_w, y_w, m_w]

    for tar_data in dataset:
        gc.collect()
        model_path = 'saved_models/DGDA_{}2{}_{}.pth'.format(args.src_data, tar_data, timestr)
        tar_ds, _, _, tar_train_loader, tar_val_loader, tar_test_loader = load_influence_dataset(
            path=args.data_path + tar_data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            seed=args.seed,
            num_workers=args.num_workers)

        val_auc_list = []
        val_f1s_list = []
        tst_auc_list = []
        tst_f1s_list = []
        for r in range(args.repeat):

            seed = 27 + r
            print('Repeat: {}/{} Seed: {}'.format(r, args.repeat, seed))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            model = DGDA(n_feat, args.enc_hidden_dim, args.dec_hidden_dim,
                         args.d_dim, args.y_dim, args.m_dim,
                         args.droprate, args.backbone,
                         src_ds.get_embedding(), src_ds.get_vertex_features(),
                         tar_ds.get_embedding(), tar_ds.get_vertex_features())
            model = model.to(device)

            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=args.lr_decay_rate)

            n_batch = max(len(src_train_loader), len(tar_train_loader))
            s_iter = iter(src_train_loader)
            t_iter = iter(tar_train_loader)
            tr_lossval = mn_lossval = best_val_auc = best_val_f1 = 0.0
            st = time.time()

            # Train
            early_stopping = EarlyStoppingF1(patience=args.patience, verbose=True, save_path=model_path)
            for batch_idx in range(args.epoch * n_batch):

                # load batch data
                s_adj, s_feats, s_labels, s_vts = next(s_iter)
                t_adj, t_feats, t_labels, t_vts = next(t_iter)

                # reset batch iterator
                if s_adj.shape[0] < args.batch_size:
                    s_iter = iter(src_train_loader)
                    s_adj, s_feats, s_labels, s_vts = next(s_iter)
                if t_adj.shape[0] < args.batch_size:
                    t_iter = iter(tar_train_loader)
                    t_adj, t_feats, t_labels, t_vts = next(t_iter)

                s_adj = s_adj.to(device)
                s_feats = s_feats.to(device)
                s_labels = s_labels.to(device)
                s_vts = s_vts.to(device)

                t_adj = t_adj.to(device)
                t_feats = t_feats.to(device)
                t_labels = t_labels.to(device)
                t_vts = t_vts.to(device)

                # train with original data
                model.train()
                optimizer.zero_grad()
                s_out = model(s_feats, s_vts, s_adj, 0)
                t_out = model(t_feats, t_vts, t_adj, 1)

                src_tr_loss = DGDA_loss(s_out, s_labels, s_adj, 0, weights)
                tar_tr_loss = DGDA_loss(t_out, t_labels, t_adj, 1, weights)
                tr_loss = src_tr_loss + tar_tr_loss
                tr_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                tr_lossval += tr_loss.item()

                # train with manipulated data
                if (batch_idx+1) % args.manipulate_batch == 0:
                    s_nadj, s_dadj = drop_edges(s_adj, args.edge_drop_rate, args.edge_add_rate)
                    t_nadj, t_dadj = drop_edges(t_adj, args.edge_drop_rate, args.edge_add_rate)
                    optimizer.zero_grad()
                    s_out = model(s_feats, s_vts, s_nadj, 0)
                    t_out = model(t_feats, t_vts, t_nadj, 1)
                    s_mn_loss = DGDA_loss(s_out, s_labels, s_nadj, 0, weights)
                    t_mn_loss = DGDA_loss(t_out, t_labels, t_nadj, 1, weights)
                    mn_loss = s_mn_loss + t_mn_loss

                    mn_loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    mn_lossval += mn_loss.item()

                # Check Loss
                if (batch_idx + 1) % args.check_batch == 0:
                    print('\nBatch={} TrainLoss={:.4f} MnpltLoss={:.4f} Time={:.2f}(s)'.format(
                        batch_idx,
                        tr_lossval / args.check_batch,
                        mn_lossval / args.check_batch,
                        time.time() - st))

                    tr_lossval = mn_lossval = 0.0
                    st = time.time()

                # Validate
                if (batch_idx + 1) % args.validate_batch == 0:
                    auc, f1 = evaluate(model, tar_val_loader, 1, device)
                    if best_val_f1 < f1:
                        best_val_f1 = f1
                        best_val_auc = auc
                    learning_rate_decay(optimizer, decay_rate=args.lr_decay_rate)
                    early_stopping(f1, model)
                    if early_stopping.early_stop:
                        print('Early stopping!')
                        break

            # Test
            print('\nTesting...')
            model.load_state_dict(torch.load(model_path))
            auc, f1 = evaluate(model, tar_test_loader, 1, device)
            logger.info('{} to {} Val  AUC: {:.6f} F1: {:.6f}'.format(args.src_data, tar_data, best_val_auc, best_val_f1))
            logger.info('{} to {} Test AUC: {:.6f} F1: {:.6f}'.format(args.src_data, tar_data, auc, f1))
            tst_auc_list.append(auc)
            tst_f1s_list.append(f1)
            val_auc_list.append(best_val_auc)
            val_f1s_list.append(best_val_f1)

        tst_auc = np.array(tst_auc_list)
        tst_f1s = np.array(tst_f1s_list)
        val_auc = np.array(val_auc_list)
        val_f1s = np.array(val_f1s_list)

        logger.info('Val & Test summary, {} to {}'.format(args.src_data, tar_data))
        logger.info('Val AUC: {:.6f}~{:.6f}'.format(val_auc.mean(), val_auc.std()))
        logger.info('Val F1:  {:.6f}~{:.6f}'.format(val_f1s.mean(), val_f1s.std()))
        logger.info('Tst AUC: {:.6f}~{:.6f}'.format(tst_auc.mean(), tst_auc.std()))
        logger.info('Tst F1:  {:.6f}~{:.6f}\n'.format(tst_f1s.mean(), tst_f1s.std()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGDA')

    parser.add_argument('--data_path', default='/data0/wfzdata/python_workspace/DAonGraph/data/')

    parser.add_argument('--src_data', default='oag',
                        help='Source domain dataset. (oag, twitter, weibo, digg)')

    parser.add_argument('--tar_data', default='weibo')
    parser.add_argument('--seed', type=int, default=27)
    parser.add_argument('--backbone', default='gcn', help='Backbone Feature Extractor GNN. gcn / gat / gin')

    parser.add_argument('--enc_hidden_dim', type=int, default=256,
                        help='Dimension of the feature extractor hidden layer. Default is 256. ')

    parser.add_argument('--d_dim', type=int, default=64,
                        help='Dimension of the domain latent variables. Default is 64. ')

    parser.add_argument('--y_dim', type=int, default=256,
                        help='Dimension of the semantic latent variables. Default is 256. ')

    parser.add_argument('--m_dim', type=int, default=128,
                        help='Dimension of the semantic latent variables. Default is 256. ')

    parser.add_argument('--dec_hidden_dim', type=int, default=64,
                        help='Dimension of the graph decoder hidden layer. Default is 64. ')

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--train_ratio', type=float, default=0.75)
    parser.add_argument('--val_ratio', type=float, default=0.125)
    parser.add_argument('--recons_weight', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--ent_weight', type=float, default=1.0)
    parser.add_argument('--d_weight', type=float, default=1.0)
    parser.add_argument('--y_weight', type=float, default=1.0)
    parser.add_argument('--m_weight', type=float, default=0.1)
    parser.add_argument('--check_batch', type=float, default=1)
    parser.add_argument('--validate_batch', type=float, default=1)
    parser.add_argument('--manipulate_batch', type=int, default=1)

    parser.add_argument('--edge_add_rate', type=float, default=0.1)
    parser.add_argument('--edge_drop_rate', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--device', type=int, default=1)

    args = parser.parse_args()

    run(args)






