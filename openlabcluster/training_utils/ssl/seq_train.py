"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from torch.nn.utils import clip_grad_norm_

# OpenLabCluster import
from openlabcluster.training_utils.ssl.utilities import save_checkpoint
from openlabcluster.training_utils.ssl.data_loader import *
from openlabcluster.user_interface.plotting_utils import format_axes
from openlabcluster.training_utils.ssl.extract_hidden import extract_hidden_ordered


DIMENSION_REDUCTION_DICT = {'PCA': PCA, 'tSNE': TSNE, 'UMAP':umap.UMAP}
def train_iter(input_tensor, seq_len, label, model, optimizer, criterion_seq, criterion_cla, alpha, device):
    """Computes training losses and propagates back to update model parameters

    Inputs:
        input_tensor: a batch of sequences (batch_size, sequence length, feature length)
        seq_len: sequence lengths of the input tensors before padding
        label: labels of sequences
        model: a random initialized model
        optimizer: the optimizer to train network
        criterion_seq: the loss function for sequence regeneration task
        criterion_cla: the loss function for classification task
        alpha: the ratio for the sequence loss (alpha) and classification (1-alpha) loss, 0<= alpha <=1
        device: str, either 'cuda' or 'cpu'
    Outputs:
        seq_loss: the loss from the regeneration task
        cla_loss: the loss from the classification task
        en_hi: the hidden states of the encoder-decoder network
        cla_pre: the predicted class labels
    """
    optimizer.zero_grad()
    if alpha == 0:
        en_hi, de_out = model(input_tensor, seq_len)
        cla_loss = 0
        cla_pre = None
    else:
        en_hi, de_out, cla_pre = model(input_tensor, seq_len)
        if sum(label != 0) != 0:
            cla_loss = criterion_cla(cla_pre[label != 0], label[label != 0] - 1)
        else:
            cla_loss = 0

    mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
    for ith_batch in range(len(seq_len)):
        mask[ith_batch, 0:seq_len[ith_batch]] = 1
    mask = torch.sum(mask, 1)

    seq_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
    seq_loss = torch.mean(torch.sum(seq_loss, 1) / mask)

    total_loss = alpha * cla_loss + (1 - alpha) * seq_loss
    if cla_loss!=0:
        cla_loss = cla_loss.item()

    total_loss.backward()
    clip_grad_norm_(model.parameters(), 25, norm_type=2)

    optimizer.step()
    del mask
    return seq_loss.item(), cla_loss, en_hi, cla_pre

def clustering_knn_acc(model, train_loader, hidden_size,
                       alpha, figure, epoch,
                       device, reducer_name, dimension
                       ):
    """Computes hidden states of keypoint sequences in the data_loader, performs dimension reduction, and plots results
        Inputs:
            model: an encoder-decoder model
            train_loader: the dataloader to load keypoint sequences
            hidden_size: the dimension of hidden states
            alpha: the ratio for the sequence loss (alpha) and classification (1-alpha) loss, 0<= alpha <=1
            figure: handle of the figure to plot hidden state
            epoch: current training epoch
            device: device where model located, either 'cuda' or 'cpu'
            reducer_name: methods for dimension reduction (options: "PCA", "tSNE", "UMAP")
            dimension: visualization dimension either '2d' or '3d'
    """
    if alpha == 0:
        hi_train, label_train,  train_semi = extract_hidden_ordered(model, train_loader, hidden_size, alpha, device)
    else:
        hi_train, label_train, train_semi,_,_ = extract_hidden_ordered(model, train_loader, hidden_size, alpha, device)
    dim_reducer = DIMENSION_REDUCTION_DICT[reducer_name]
    if dimension == '2d':
        reducer = dim_reducer(n_components=2)
    else:
        reducer = dim_reducer(n_components=3)
    transformed = reducer.fit_transform(hi_train)
    if figure:
        figure.axes.cla()
        if dimension == '2d':
            figure.axes.scatter(transformed[:,0], transformed[:,1],s=10, picker=True, color='k')
        else:
            figure.axes.scatter(transformed[:, 0], transformed[:, 1],  transformed[:, 1], s=10, picker=True, color='k')
        if alpha == 0:
            figure.axes.set_title('Cluster Map Epoch %d' % epoch)
        else:
            figure.axes.set_title('Behavior Classification Map Epoch %d' % epoch)

        format_axes(axes=figure.axes)
        figure.canvas.draw_idle()


def training(ith_epoch, epoch, train_loader,
             model, optimizer, criterion_seq, criterion_cla, alpha, past_loss,
             model_path, pre,  model_prefix,
             device='cuda'):
    """Wrapper of train_iter performing model training
    Inputs:
        ith_epoch: int, current training epoch
        epoch: the maximum number of training epoch
        train_loader: the dataloader to load keypoint sequences
        model: a sequence to sequence model
        optimizer: optimizer to train network
        criterion_seq: the loss function for the sequence regeneration task
        criterion_cla: the loss function for the classification task
        alpha: the ratio for the sequence loss (alpha) and classification (1-alpha) loss, 0<= alpha <=1
        past_loss: the minimum training loss during the training
        model_path: the directory where previous model was saved
        pre: the prefix of models to overwrite
        model_prefix: the prefix of the model to save
        device: str, either 'cuda' or 'cpu'
    Outputs:
        past_loss: the updated minimum loss
        path_model: the directory of the saved model
        acc(optional): the classification accuracy if alpha>0
    """
    cla = 0
    seq=0
    corr_num = 0
    labeled_num = 0
    for it, (data, seq_len, label, semi_label,_) in enumerate(train_loader):
        input_tensor = data.to(device)
        semi_label = torch.tensor(semi_label, dtype=torch.long).to(device)
        seq_loss, cla_loss, en_hid, cla_pre = train_iter(input_tensor, seq_len, semi_label, model, optimizer, criterion_seq,
                                                 criterion_cla, alpha, device)
        if alpha >0:
            corr_num += sum((torch.argmax(cla_pre, axis=1)[semi_label!=0]+1) == semi_label[semi_label!=0])
        labeled_num += sum(semi_label!=0)
        cla += cla_loss
        seq += seq_loss
        loss = (1-alpha)*seq + alpha*cla
    print(f"Clas loss: {cla/(it+1):.3f} seq_loss:{seq/(it+1):.3f} acc {corr_num/labeled_num}")

    if loss < past_loss:
        past_loss = loss
        if os.path.exists(model_path):
            for item in os.listdir(model_path):
                if item.startswith(pre):
                    if os.path.exists('./models'):
                        # Overwrites and make the file blank instead
                        open('./models/' + item, 'w').close()
                        os.remove('./models/' + item)
        else:
            os.mkdir(model_path)
        path_model = model_prefix + '_epoch%d' % (ith_epoch)

        save_checkpoint(model, epoch, optimizer, loss, path_model)
    else:
        path_model = None
    acc = corr_num/labeled_num
    if alpha > 0:
        return past_loss, path_model, acc.cpu().numpy().item()
    else:
        return past_loss, path_model




