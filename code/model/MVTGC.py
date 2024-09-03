import math
import datetime
import torch
from torch.autograd import Variable
# torch.autograd.Variable是Autograd的核心类，它封装了Tensor，并整合了反向传播的相关实现
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.cluster import KMeans
import numpy as np
import sys
from model.DataSet import MVTGCDataSet
from model.evaluation import eva
from torch.nn import Linear
import torch.nn.functional as F

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0


class MVTGC:
    def __init__(self, args):
        self.args = args
        self.the_data = args.dataset
        self.file_path = '../data/%s/%s.txt' % (self.the_data, self.the_data)
        self.emb_path = '../emb/%s/%s_MVTGC.emb'
        self.label_path = '../data/%s/label.txt' % (self.the_data)
        self.labels = self.read_label()
        self.r_RW = args.r_RW
        self.r_PE = args.r_PE

        self.emb_size = args.emb_size
        self.neg_size = args.neg_size
        self.hist_len = args.hist_len
        self.batch = args.batch_size
        self.clusters = args.clusters
        self.save_step = args.save_step
        self.epochs = args.epoch
        self.best_acc = 0
        self.best_nmi = 0
        self.best_ari = 0
        self.best_f1 = 0
        self.best_epoch = 0

        self.data = MVTGCDataSet(args, self.file_path, self.neg_size, self.hist_len, args.directed)
        self.node_dim = self.data.get_node_dim()
        self.feature_RW = self.data.get_View_RW()
        self.feature_PE = self.data.get_View_PE()
        self.feature_A = self.data.get_View_A()
        if self.the_data == 'arxivAI':
            self.feature_A = np.concatenate((self.feature_A, self.feature_A, self.feature_A, self.feature_A,
                                             self.feature_A, self.feature_A, self.feature_A, self.feature_A), axis=1)
        self.loss_fn = torch.nn.MSELoss()
        self.similarity = torch.nn.CosineSimilarity(dim=2)
        self.feature_fusion = self.feature_RW * self.r_RW + self.feature_PE * self.r_PE + self.feature_A * (1 - self.r_RW - self.r_PE)
        self.main_feature = self.feature_fusion

        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.node_emb = Variable(torch.from_numpy(self.main_feature).type(FType).cuda(), requires_grad=True)
                self.View_RW = Variable(torch.from_numpy(self.feature_RW).type(FType).cuda(), requires_grad=False)
                self.View_PE = Variable(torch.from_numpy(self.feature_PE).type(FType).cuda(), requires_grad=False)
                self.View_A = Variable(torch.from_numpy(self.feature_A).type(FType).cuda(), requires_grad=False)

                self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.cluster_layer = Variable((torch.zeros(self.clusters, self.emb_size) + 1.).type(FType).cuda(), requires_grad=True)
                torch.nn.init.xavier_normal_(self.cluster_layer.data)

                kmeans = KMeans(n_clusters=self.clusters, n_init=20)
                _ = kmeans.fit_predict(self.main_feature)
                self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()
                self.v = 1.0

        self.opt = SGD(lr=args.learning_rate, params=[self.node_emb, self.delta, self.cluster_layer])
        self.loss = torch.FloatTensor()

    def read_label(self):
        labels = []
        with open(self.label_path, 'r') as reader:
            for line in reader:
                label = int(line)
                labels.append(label)
        return labels

    def kl_loss(self, z, p):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        the_kl_loss = F.kl_div((q.log()), p, reduction='batchmean')  # l_clu
        return the_kl_loss

    def target_dis(self, emb):
        q = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        tmp_q = q.data
        weight = tmp_q ** 2 / tmp_q.sum(0)
        p = (weight.t() / weight.sum(1)).t()

        return p

    def contrastive_loss(self, view1_feature, view2_feature):
        sim_view12 = self.similarity(view1_feature.unsqueeze(1), view2_feature.unsqueeze(0))

        logits_view12 = sim_view12 - torch.log(torch.exp(1.06 * sim_view12).sum(1, keepdim=True))
        logits_view21 = sim_view12.T - torch.log(torch.exp(1.06 * sim_view12.T).sum(1, keepdim=True))

        loss = - torch.diag(logits_view12).mean() - torch.diag(logits_view21).mean()

        return loss

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):

        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        s_View_RW = self.View_RW.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        s_View_PE = self.View_PE.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        s_View_A = self.View_A.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)

        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(dim=1)  # [b]

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)

        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                loss = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(
                    n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)  # [b]
        else:
            loss = -torch.log(torch.sigmoid(p_lambda) + 1e-6) - torch.log(
                torch.sigmoid(torch.neg(n_lambda)) + 1e-6).sum(dim=1)

        l_x = torch.norm(s_node_emb - s_View_RW, p=2) * self.r_RW + torch.norm(s_node_emb - s_View_PE, p=2) * self.r_PE\
              + torch.norm(s_node_emb - s_View_A, p=2) * (1 - self.r_RW - self.r_PE) + 1e-6

        p_View_RW = self.target_dis(s_View_RW)
        p_View_PE = self.target_dis(s_View_PE)
        p_View_A = self.target_dis(s_View_A)
        
        l_d = self.kl_loss(s_node_emb, p_View_RW) * self.r_RW + self.kl_loss(s_node_emb, p_View_PE) * self.r_PE\
              + self.kl_loss(s_node_emb, p_View_A) * (1 - self.r_RW - self.r_PE)

        l_framework = l_d + l_x

        if self.the_data == 'school':
            total_loss = l_framework
        elif self.the_data == 'arxivAI':
            total_loss = loss.sum() + l_d
        else:
            total_loss = loss.sum() + l_framework

        return total_loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                self.loss += loss.data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            start = datetime.datetime.now()
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=4)

            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                if torch.cuda.is_available():
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['target_time'].type(FType).cuda(),
                                    sample_batched['neg_nodes'].type(LType).cuda(),
                                    sample_batched['history_nodes'].type(LType).cuda(),
                                    sample_batched['history_times'].type(FType).cuda(),
                                    sample_batched['history_masks'].type(FType).cuda())
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType))

            acc, nmi, ari, f1 = eva(self.clusters, self.labels, self.node_emb)

            if acc > self.best_acc:
                self.best_acc = acc
                self.best_nmi = nmi
                self.best_ari = ari
                self.best_f1 = f1
                self.best_epoch = epoch
                self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data))

            sys.stdout.write('\repoch %d: loss=%.4f  ' % (epoch, (self.loss.cpu().numpy() / len(self.data))))
            sys.stdout.write('ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)\n' % (acc, nmi, ari, f1))

            end = datetime.datetime.now()
            print('Training Complete with Time: %s' % str(end - start))

            sys.stdout.flush()

        print('Best performance in %d epoch: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
              (self.best_epoch, self.best_acc, self.best_nmi, self.best_ari, self.best_f1))


    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()
