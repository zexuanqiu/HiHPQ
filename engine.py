import os
import logging
from collections import OrderedDict
from evaluation import get_db_codes_and_targets
from tqdm import tqdm
import numpy as np
import torch
from network import HyperPQ
import tensorboard
from hierarchical_funcs import compute_features, hier_clus
from loss import *
import time 

def train(datahub, model: HyperPQ, loss_fn1, loss_fn2: ProtoLoss,  optimizer, lr_scheduler, config, 
          compute_err=True, evaluator=None, monitor=None, writer=None, save_quant_error=None,
          prot_loss_weight=1.0, neighbor_loss_weight=0.1):
    model = model.to(config.device)

    all_quant_error = []

    cluster_result = None
    im2cluster = None
    for epoch in range(config.epoch_num):
        # compute clusters before every training
        all_feats = []
        all_hat_feats = []

        if epoch > config.warmup_epoch and (epoch - config.warmup_epoch) % config.clus_interval == 0:
            logging.info('Current epoch is :{}'.format(epoch))
            start = time.time()
            features = compute_features(datahub.clus_loader, model).cpu().numpy()

            cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
            for num_cluster in config.num_clus_list:
                cluster_result['im2cluster'].append(torch.zeros(len(datahub.clus_loader.dataset),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster), config.feat_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda()) 
                
            if config.clus_mode == "hier_clus":
                cluster_result = hier_clus(x=features, num_clus_list=config.num_clus_list)
            else:
                raise not NotImplementedError("")
            clus_end = time.time()
            logging.info("Hierarchical Clustering takes {}s".format(clus_end - start))

            if cluster_result is not None:
                im2cluster = cluster_result['im2cluster'] 


        epoch_loss, epoch_aug_inst_loss, epoch_neighbor_inst_loss, epoch_prot_loss, epoch_quant_err, batch_num = 0, 0, 0, 0, 0,  len(datahub.train_loader)
        
        for i, (train_data, index, _) in enumerate(tqdm(datahub.train_loader, desc="epoch %d" % epoch)):
            global_step = i + epoch * batch_num
            view1_data = train_data[0].to(config.device)
            view2_data = train_data[1].to(config.device)
            if lr_scheduler is not None:
                curr_lr = lr_scheduler.step()
                if writer is not None:
                    writer.add_scalar('lr', curr_lr, global_step)
            optimizer.zero_grad()
            
            # forward data and produce features and codes for 2 views
            view1_feats, view1_hat_feats, view1_soft_codes, view1_err = model(view1_data)
            # softsort_cwd1 [B,K,D,M]
            view2_feats, view2_hat_feats, view2_soft_codes, view2_err = model(view2_data)

            all_feats.append(view1_feats.detach().cpu())
            all_hat_feats.append(view1_hat_feats.detach().cpu())
            

            if im2cluster is not None:
                tmp = [i_im2cluster[index] for i_im2cluster in im2cluster]
                aug_inst_loss, neighbor_inst_loss = loss_fn1(view1_hat_feats, view2_hat_feats, model.hyper_pq_head.neg_curvs, im2cluster=tmp)
            else:
                aug_inst_loss, neighbor_inst_loss = loss_fn1(view1_hat_feats, view2_hat_feats, model.hyper_pq_head.neg_curvs, im2cluster=None)
                    

            if writer is not None:
                writer.add_scalar('loss/hyper_aug_inst_loss', aug_inst_loss.item(), global_step)
                writer.add_scalar('loss/hyper_neighbor_inst_loss', neighbor_inst_loss.item(), global_step)
            
            # prototypical-constrastive loss if needed
            if cluster_result is not None:
                proto_loss_list = loss_fn2(view1_hat_feats=view1_hat_feats, neg_curvs=model.hyper_pq_head.neg_curvs,
                                           tangent_to_hyper_func=model.hyper_pq_head.tangent_to_hyper,
                                           clus_mode=config.clus_mode, index=index, cluster_result=cluster_result)
                proto_loss = sum(proto_loss_list) / float(len(proto_loss_list))
                if writer is not None:
                    for step, cur_loss in enumerate(proto_loss_list):
                        writer.add_scalar("'loss/hyper_proto_loss_{}".format(step), cur_loss.item(), global_step)

                loss = 1.0 * aug_inst_loss +  prot_loss_weight * proto_loss + neighbor_loss_weight * neighbor_inst_loss
                        
            else:
                proto_loss = 0.
                loss = aug_inst_loss + neighbor_inst_loss

            if writer is not None:
                writer.add_scalar('loss/all_loss', loss.item(), global_step)
            
            if compute_err:
                quant_err = (view1_err + view2_err) / 2
                epoch_quant_err += quant_err.item()
                all_quant_error.append(quant_err.item())
                if writer is not None:
                    writer.add_scalar('quant_err', quant_err.item(), global_step)
                
            epoch_loss += loss.item()
            epoch_aug_inst_loss += aug_inst_loss.item()
            epoch_neighbor_inst_loss += neighbor_inst_loss.item()
            if proto_loss > 0: epoch_prot_loss += proto_loss.item()

            loss.backward()
            optimizer.step()

            # keep the neg_curvs > 0
            model.hyper_pq_head.neg_curvs.data.clamp_(min=0.01, max=10)

        logging.info("epoch %d: avg loss=%f,  avg aug inst loss=%f, avg neighbor inst loss=%f, avg proto loss=%f, avg quantization error=%f" % 
                     (epoch, epoch_loss / batch_num, epoch_aug_inst_loss / batch_num, epoch_neighbor_inst_loss / batch_num,  epoch_prot_loss / batch_num, epoch_quant_err / batch_num))
        if model.hyper_pq_head.alpha_scaler is not None:
            logging.info("epoch {}, --alpha_scaler {}".format(epoch, model.hyper_pq_head.alpha_scaler))

        if evaluator is not None:
            if (epoch+1) % config.eval_interval == 0:
                logging.info("begin to evaluate model")

                hyper_codebooks = model.hyper_codebooks()
                evaluator.set_codebooks(codebooks=hyper_codebooks)
                db_codes, db_targets = get_db_codes_and_targets(datahub.database_loader, 
                                                                model, device=config.device)

                evaluator.set_db_codes(db_codes=db_codes)
                evaluator.set_db_targets(db_targets=db_targets)
                logging.info("compute mAP")

                val_mAP = evaluator.MAP(datahub.test_loader, model, topK=config.topK)
                logging.info("val mAP=%f" % val_mAP)
                if writer is not None:
                    writer.add_scalar("val_mAP", val_mAP, epoch)
                if monitor:
                    is_break, is_lose_patience = monitor.update(val_mAP)
                    if is_break:
                        logging.info("early stop")
                        break

    logging.info("finish trainning at epoch %d" % epoch)
    logging.info("Best Map: {}".format(monitor.best_value))

    
    if save_quant_error is not None:
        all_quant_error = np.array(all_quant_error)
        np.save(save_quant_error, all_quant_error)
    


def test(datahub, model, config, evaluator, writer=None):
    '''evaluator must be loaded with correct codebook, db_codes and db_targets'''
    logging.info("compute mAP")
    model = model.to(config.device)
    test_mAP = evaluator.MAP(datahub.test_loader, model, topK=config.topK)
    logging.info("test mAP=%f" % test_mAP)

    logging.info("finish testing")
 