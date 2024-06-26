import torch
import logging
from tqdm import tqdm
from network import HyperPQ
# import faiss 
import numpy as np 
from scipy.cluster.hierarchy import linkage, fcluster

def compute_features(train_loader, model: HyperPQ):
    logging.info("Computing features...")
    model.eval()
    features = []
    for i, (orig_img, _, _) in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            orig_img = orig_img.cuda()
            feat = model.encode_tangent_feats(orig_img)
            features.append(feat)
    features = torch.cat(features, dim=0) # (len_of_loader, dim)
    logging.info("Shape of features used for clustering is: {}".format(features.shape))
    return features.cpu()


def hier_clus(x: np.array, num_clus_list):
    results = {'im2cluster': [], 'centroids': []}
    Z = linkage(x, method="ward")
    for num_clus in num_clus_list:
        num_clus = int(num_clus)
        labels = fcluster(Z, num_clus, criterion="maxclust") -1
        centroids = np.array([x[labels == i].mean(axis=0) for i in range(0, num_clus)])    

        im2cluster = torch.LongTensor(labels).cuda()
        centroids = torch.Tensor(centroids).cuda()
        results['im2cluster'].append(im2cluster)
        results['centroids'].append(centroids)
    return results



