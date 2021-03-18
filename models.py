from layers import *
from utils import *


# DGDA components
class GNN_VGAE_Encoder(nn.Module):
    def __init__(self, in_dim, hs, dim_d, dim_y, dim_m, droprate, backbone='gcn'):
        super(GNN_VGAE_Encoder, self).__init__()
        self.backbone = backbone
        if backbone == 'gcn':
            self.gnn0 = BatchGraphConvolution(in_dim, hs)
            self.gnn1 = BatchGraphConvolution(hs, hs)
            self.d_gnn2 = BatchGraphConvolution(hs, 2 * dim_d)
            self.y_gnn2 = BatchGraphConvolution(hs, 2 * dim_y)
            self.m_gnn2 = BatchGraphConvolution(hs, 2 * dim_m)
        elif backbone == 'gat':
            self.gnn0 = BatchMultiHeadGraphAttention(1, in_dim, hs, 0.2)
            self.gnn1 = BatchMultiHeadGraphAttention(1, hs, hs, 0.2)
            self.d_gnn2 = BatchMultiHeadGraphAttention(1, hs, 2 * dim_d, 0.2)
            self.y_gnn2 = BatchMultiHeadGraphAttention(1, hs, 2 * dim_y, 0.2)
            self.m_gnn2 = BatchMultiHeadGraphAttention(1, hs, 2 * dim_m, 0.2)
        elif backbone == 'gin':
            self.gnn0 = BatchGIN(in_dim, hs, hs)
            self.gnn1 = BatchGIN(hs, hs, hs)
            self.d_gnn2 = BatchGIN(hs, hs, 2 * dim_d)
            self.y_gnn2 = BatchGIN(hs, hs, 2 * dim_y)
            self.m_gnn2 = BatchGIN(hs, hs, 2 * dim_m)
        else:
            raise NotImplementedError

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(droprate)

    def repara(self, mu, lv):
        if self.training:
            eps = torch.randn_like(lv)
            std = torch.exp(lv)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, adj):
        if self.backbone == 'gcn':
            adj = vectorized_sym_norm(adj)
        res = dict()
        h = self.dropout(self.act(self.gnn0(x, adj)))
        h = self.dropout(self.act(self.gnn1(h, adj)))
        d = self.d_gnn2(h, adj)
        y = self.y_gnn2(h, adj)
        m = self.m_gnn2(h, adj)
        res['dmu'], res['dlv'] = d.chunk(chunks=2, dim=-1)
        res['ymu'], res['ylv'] = y.chunk(chunks=2, dim=-1)
        res['mmu'], res['mlv'] = m.chunk(chunks=2, dim=-1)
        res['d'] = self.repara(res['dmu'], res['dlv'])
        res['y'] = self.repara(res['ymu'], res['ylv'])
        res['m'] = self.repara(res['mmu'], res['mlv'])
        return res, h

class GNN_Encoder(nn.Module):
    def __init__(self, in_dim, hs, dp):
        super(GNN_Encoder, self).__init__()
        self.gnn0 = BatchGraphConvolution(in_dim, hs)
        self.gnn1 = BatchGraphConvolution(hs, hs)
        self.gnn2 = BatchGraphConvolution(hs, hs)
        self.dropout = nn.Dropout(dp)
        self.act = nn.ReLU()

    def forward(self, x, adj):
        adj = vectorized_sym_norm(adj)
        h = self.dropout(self.act(self.gnn0(x, adj)))
        h = self.dropout(self.act(self.gnn1(h, adj)))
        h = self.gnn2(h, adj)

        return h

class DSR_Encoder(nn.Module):
    def __init__(self, in_dim, hs, dp):
        super(DSR_Encoder, self).__init__()
        self.gcn0 = BatchGraphConvolution(in_dim, hs)
        self.gcn1 = BatchGraphConvolution(hs, hs)
        self.sem_gcn2 = BatchGraphConvolution(hs, hs + hs)
        self.dom_gcn2 = BatchGraphConvolution(hs, hs + hs)
        self.dropout = nn.Dropout(dp)
        self.act = nn.ReLU()

    def repara(self, mu, lv):
        if self.training:
            eps = torch.randn_like(lv)
            std = torch.exp(lv)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, adj):
        res = dict()
        adj = vectorized_sym_norm(adj)
        h = self.dropout(self.act(self.gcn0(x, adj)))
        h = self.dropout(self.act(self.gcn1(h, adj)))
        d = self.dom_gcn2(h, adj)
        y = self.sem_gcn2(h, adj)
        res['dmu'], res['dlv'] = d.chunk(chunks=2, dim=-1)
        res['ymu'], res['ylv'] = y.chunk(chunks=2, dim=-1)
        res['d'] = self.repara(res['dmu'], res['dlv'])
        res['y'] = self.repara(res['ymu'], res['ylv'])
        return res

class DSR_Decoder(nn.Module):
    def __init__(self, dec_hs, dim_d, dim_y, droprate):
        super(DSR_Decoder, self).__init__()
        self.d_lin0 = nn.Linear(dim_d, dim_d)
        self.y_lin0 = nn.Linear(dim_y, dim_y)
        self.dy_lin1 = nn.Linear(dim_d + dim_y, dec_hs)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, d, y):
        d = self.dropout(self.act(self.d_lin0(d)))
        y = self.dropout(self.act(self.y_lin0(y)))
        dy = torch.cat([d, y], dim=-1)
        dy = self.dy_lin1(dy)
        adj_recons = torch.bmm(dy, dy.permute(0, 2, 1))
        return adj_recons

class GraphDecoder(nn.Module):
    def __init__(self, dec_hs, dim_d, dim_y, dim_m, droprate):
        super(GraphDecoder, self).__init__()
        self.d_lin0 = nn.Linear(dim_d, dim_d)
        self.y_lin0 = nn.Linear(dim_y, dim_y)
        self.m_lin0 = nn.Linear(dim_m, dim_m)
        self.dym_lin1 = nn.Linear(dim_d + dim_y + dim_m, dec_hs)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, d, y, m):
        d = self.dropout(self.act(self.d_lin0(d)))
        y = self.dropout(self.act(self.y_lin0(y)))
        m = self.dropout(self.act(self.m_lin0(m)))
        dym = torch.cat([d, y, m], dim=-1)
        dym = self.dym_lin1(dym)
        adj_recons = torch.bmm(dym, dym.permute(0, 2, 1))
        return adj_recons

class NoiseDecoder(nn.Module):
    def __init__(self, dim_m, droprate):
        super(NoiseDecoder, self).__init__()
        self.m_lin0 = nn.Linear(dim_m, dim_m)
        self.m_lin1 = nn.Linear(dim_m, dim_m)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.dropout(self.act(self.m_lin0(x)))
        h = self.m_lin1(h)
        noise_recons = torch.bmm(h, h.permute(0, 2, 1))
        return noise_recons

class ClassClassifier(nn.Module):
    def __init__(self, hs, n_class, droprate):
        super(ClassClassifier, self).__init__()
        self.lin0 = nn.Linear(hs, hs)
        self.lin1 = nn.Linear(hs, n_class)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.dropout(self.act(self.lin0(x)))
        logits = self.lin1(h)
        return logits[:, -1, :]  # ego-user = the last user

class DomainClassifier(nn.Module):
    def __init__(self, dim_d):
        super(DomainClassifier, self).__init__()
        self.lin = nn.Linear(dim_d, 1)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        logits = self.lin(x)
        return logits

class GraphDiscriminator(nn.Module):
    def __init__(self, in_dim, hs, droprate):
        super(GraphDiscriminator, self).__init__()
        self.lin_0 = nn.Linear(in_dim, hs)
        self.lin_1 = nn.Linear(hs, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(droprate)

    def forward(self, x):
        h = self.act(self.dropout(self.lin_0(x)))
        h = torch.mean(h, dim=1)
        logits = self.lin_1(h)
        return logits

class DGDA(nn.Module):
    def __init__(self, in_dim, enc_hs, dec_hs, dim_d, dim_y, dim_m, droprate, backbone,
                 source_pretrained_emb, source_vertex_feats,
                 target_pretrained_emb, target_vertex_feats):
        super(DGDA, self).__init__()
        self.semb = nn.Embedding(source_pretrained_emb.size(0), source_pretrained_emb.size(1))
        self.semb.weight = Parameter(source_pretrained_emb, requires_grad=False)
        self.temb = nn.Embedding(target_pretrained_emb.size(0), target_pretrained_emb.size(1))
        self.temb.weight = Parameter(target_pretrained_emb, requires_grad=False)
        in_dim += int(source_pretrained_emb.size(1))

        self.svf = nn.Embedding(source_vertex_feats.size(0), source_vertex_feats.size(1))
        self.svf.weight = Parameter(source_vertex_feats, requires_grad = False)
        self.tvf = nn.Embedding(target_vertex_feats.size(0), target_vertex_feats.size(1))
        self.tvf.weight = Parameter(target_vertex_feats, requires_grad = False)
        in_dim += int(source_vertex_feats.size(1))

        self.encoder = GNN_VGAE_Encoder(in_dim, enc_hs, dim_d, dim_y, dim_m, droprate, backbone)
        self.graphDiscriminator = GraphDiscriminator(enc_hs, enc_hs//2, droprate)
        self.graph_decoder = GraphDecoder(dec_hs, dim_d, dim_y, dim_m, droprate)
        self.noise_decoder = NoiseDecoder(dim_m, droprate)
        self.classClassifier = ClassClassifier(dim_y, 2, droprate)
        self.domainClassifier = DomainClassifier(dim_d)

    def forward(self, x, vts, adj, domain, recon=True):
        if domain == 0:
            x = torch.cat((x, self.semb(vts)), dim=-1)
            x = torch.cat((x, self.svf(vts)), dim=-1)
        else:
            x = torch.cat((x, self.temb(vts)), dim=-1)
            x = torch.cat((x, self.tvf(vts)), dim=-1)


        x = F.instance_norm(x)
        res, h = self.encoder(x, adj)

        if recon:
            res['a_recons'] = self.graph_decoder(res['d'], res['y'], res['m'])
            res['m_recons'] = self.noise_decoder(res['m'])
        res['dom_output'] = self.domainClassifier(res['d'])
        res['cls_output'] = self.classClassifier(res['y'])

        return res

class GraphDecoder_m(nn.Module):
    def __init__(self, dec_hs, dim_d, dim_y, droprate):
        super(GraphDecoder_m, self).__init__()
        self.d_lin0 = nn.Linear(dim_d, dim_d)
        self.y_lin0 = nn.Linear(dim_y, dim_y)
        self.dym_lin1 = nn.Linear(dim_d + dim_y, dec_hs)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, d, y):
        d = self.dropout(self.act(self.d_lin0(d)))
        y = self.dropout(self.act(self.y_lin0(y)))
        dy = torch.cat([d, y], dim=-1)
        dy = self.dym_lin1(dy)
        adj_recons = torch.bmm(dy, dy.permute(0, 2, 1))
        return adj_recons

class DGDA_m(nn.Module):
    def __init__(self, in_dim, enc_hs, dec_hs, dim_d, dim_y, dim_m, droprate, backbone,
                 source_pretrained_emb, source_vertex_feats,
                 target_pretrained_emb, target_vertex_feats):
        super(DGDA_m, self).__init__()
        self.semb = nn.Embedding(source_pretrained_emb.size(0), source_pretrained_emb.size(1))
        self.semb.weight = Parameter(source_pretrained_emb, requires_grad=False)
        self.temb = nn.Embedding(target_pretrained_emb.size(0), target_pretrained_emb.size(1))
        self.temb.weight = Parameter(target_pretrained_emb, requires_grad=False)
        in_dim += int(source_pretrained_emb.size(1))

        self.svf = nn.Embedding(source_vertex_feats.size(0), source_vertex_feats.size(1))
        self.svf.weight = Parameter(source_vertex_feats, requires_grad = False)
        self.tvf = nn.Embedding(target_vertex_feats.size(0), target_vertex_feats.size(1))
        self.tvf.weight = Parameter(target_vertex_feats, requires_grad = False)
        in_dim += int(source_vertex_feats.size(1))

        self.encoder = GNN_VGAE_Encoder(in_dim, enc_hs, dim_d, dim_y, dim_m, droprate, backbone)
        self.graphDiscriminator = GraphDiscriminator(enc_hs, enc_hs//2, droprate)
        self.graph_decoder = GraphDecoder_m(dec_hs, dim_d, dim_y, droprate)
        self.classClassifier = ClassClassifier(dim_y, 2, droprate)
        self.domainClassifier = DomainClassifier(dim_d)

    def forward(self, x, vts, adj, domain, recon=True):
        if domain == 0:
            x = torch.cat((x, self.semb(vts)), dim=-1)
            x = torch.cat((x, self.svf(vts)), dim=-1)
        else:
            x = torch.cat((x, self.temb(vts)), dim=-1)
            x = torch.cat((x, self.tvf(vts)), dim=-1)


        x = F.instance_norm(x)
        res, h = self.encoder(x, adj)

        if recon:
            res['a_recons'] = self.graph_decoder(res['d'], res['y'])
        res['dom_output'] = self.domainClassifier(res['d'])
        res['cls_output'] = self.classClassifier(res['y'])

        return res

# Traditional DA methods
class DANN(nn.Module):
    def __init__(self, in_dim, hs, dp,
                 source_pretrained_emb, source_vertex_feats,
                 target_pretrained_emb, target_vertex_feats):
        super(DANN, self).__init__()
        self.semb = nn.Embedding(source_pretrained_emb.size(0), source_pretrained_emb.size(1))
        self.semb.weight = Parameter(source_pretrained_emb, requires_grad=False)
        self.temb = nn.Embedding(target_pretrained_emb.size(0), target_pretrained_emb.size(1))
        self.temb.weight = Parameter(target_pretrained_emb, requires_grad=False)
        in_dim += int(source_pretrained_emb.size(1))

        self.svf = nn.Embedding(source_vertex_feats.size(0), source_vertex_feats.size(1))
        self.svf.weight = Parameter(source_vertex_feats, requires_grad = False)
        self.tvf = nn.Embedding(target_vertex_feats.size(0), target_vertex_feats.size(1))
        self.tvf.weight = Parameter(target_vertex_feats, requires_grad = False)
        in_dim += int(source_vertex_feats.size(1))

        self.encoder = GNN_Encoder(in_dim, hs, dp)
        self.classClassifier = ClassClassifier(hs, 2, dp)
        self.domainClassifier = DomainClassifier(hs)

    def forward(self, x, vts, adj, domain, grl_lamda=0.1):
        res = dict()
        if domain == 0:
            x = torch.cat((x, self.semb(vts)), dim=-1)
            x = torch.cat((x, self.svf(vts)), dim=-1)
        else:
            x = torch.cat((x, self.temb(vts)), dim=-1)
            x = torch.cat((x, self.tvf(vts)), dim=-1)

        x = F.instance_norm(x)
        h = self.encoder(x, adj)
        res['cls_output'] = self.classClassifier(h)
        res['dom_output'] = self.domainClassifier(grad_reverse(h, grl_lamda))
        return res

class MDD(nn.Module):
    def __init__(self, in_dim, hs, dp,
                 source_pretrained_emb, source_vertex_feats,
                 target_pretrained_emb, target_vertex_feats):
        super(MDD, self).__init__()

        self.semb = nn.Embedding(source_pretrained_emb.size(0), source_pretrained_emb.size(1))
        self.semb.weight = Parameter(source_pretrained_emb, requires_grad=False)
        self.temb = nn.Embedding(target_pretrained_emb.size(0), target_pretrained_emb.size(1))
        self.temb.weight = Parameter(target_pretrained_emb, requires_grad=False)
        in_dim += int(source_pretrained_emb.size(1))

        self.svf = nn.Embedding(source_vertex_feats.size(0), source_vertex_feats.size(1))
        self.svf.weight = Parameter(source_vertex_feats, requires_grad = False)
        self.tvf = nn.Embedding(target_vertex_feats.size(0), target_vertex_feats.size(1))
        self.tvf.weight = Parameter(target_vertex_feats, requires_grad = False)
        in_dim += int(source_vertex_feats.size(1))

        self.encoder = GNN_Encoder(in_dim, hs, dp)
        self.classClassifier = ClassClassifier(hs, 2, dp)
        self.adv_classClassifier = ClassClassifier(hs, 2, dp)

    def forward(self, x, vts, adj, domain, grl_lamda=0.1):
        res = dict()
        if domain == 0:
            x = torch.cat((x, self.semb(vts)), dim=-1)
            x = torch.cat((x, self.svf(vts)), dim=-1)
        else:
            x = torch.cat((x, self.temb(vts)), dim=-1)
            x = torch.cat((x, self.tvf(vts)), dim=-1)

        x = F.instance_norm(x)
        h = self.encoder(x, adj)
        res['cls_output'] = self.classClassifier(h)
        res['adv_output'] = self.adv_classClassifier(grad_reverse(h, grl_lamda))
        return res

# Disentanglement-based models
class DIVA(nn.Module):
    def __init__(self, in_dim, enc_hs, dec_hs, dim_d, dim_y, dim_m, dp,
                 source_pretrained_emb, source_vertex_feats,
                 target_pretrained_emb, target_vertex_feats):
        super(DIVA, self).__init__()
        self.semb = nn.Embedding(source_pretrained_emb.size(0), source_pretrained_emb.size(1))
        self.semb.weight = Parameter(source_pretrained_emb, requires_grad=False)
        self.temb = nn.Embedding(target_pretrained_emb.size(0), target_pretrained_emb.size(1))
        self.temb.weight = Parameter(target_pretrained_emb, requires_grad=False)
        in_dim += int(source_pretrained_emb.size(1))

        self.svf = nn.Embedding(source_vertex_feats.size(0), source_vertex_feats.size(1))
        self.svf.weight = Parameter(source_vertex_feats, requires_grad = False)
        self.tvf = nn.Embedding(target_vertex_feats.size(0), target_vertex_feats.size(1))
        self.tvf.weight = Parameter(target_vertex_feats, requires_grad = False)
        in_dim += int(source_vertex_feats.size(1))

        self.encoder = GNN_VGAE_Encoder(in_dim, enc_hs, dim_d, dim_y, dim_m, dp)
        self.graph_decoder = GraphDecoder(dec_hs, dim_d, dim_y, dim_m, dp)
        self.classClassifier = ClassClassifier(dim_y, 2, dp)
        self.domainClassifier = DomainClassifier(dim_d)

    def forward(self, x, vts, adj, domain, recon=True):
        if domain == 0:
            x = torch.cat((x, self.semb(vts)), dim=-1)
            x = torch.cat((x, self.svf(vts)), dim=-1)
        else:
            x = torch.cat((x, self.temb(vts)), dim=-1)
            x = torch.cat((x, self.tvf(vts)), dim=-1)

        x = F.instance_norm(x)
        res, h = self.encoder(x, adj)
        if recon:
            res['a_recons'] = self.graph_decoder(res['d'], res['y'], res['m'])
        res['dom_output'] = self.domainClassifier(res['d'])
        res['cls_output'] = self.classClassifier(res['y'])

        return res

class DSR(nn.Module):
    def __init__(self, in_dim, hs, droprate,
                 source_pretrained_emb, source_vertex_feats,
                 target_pretrained_emb, target_vertex_feats):
        super(DSR, self).__init__()
        self.semb = nn.Embedding(source_pretrained_emb.size(0), source_pretrained_emb.size(1))
        self.semb.weight = Parameter(source_pretrained_emb, requires_grad=False)
        self.temb = nn.Embedding(target_pretrained_emb.size(0), target_pretrained_emb.size(1))
        self.temb.weight = Parameter(target_pretrained_emb, requires_grad=False)
        in_dim += int(source_pretrained_emb.size(1))

        self.svf = nn.Embedding(source_vertex_feats.size(0), source_vertex_feats.size(1))
        self.svf.weight = Parameter(source_vertex_feats, requires_grad = False)
        self.tvf = nn.Embedding(target_vertex_feats.size(0), target_vertex_feats.size(1))
        self.tvf.weight = Parameter(target_vertex_feats, requires_grad = False)
        in_dim += int(source_vertex_feats.size(1))

        self.encoder = DSR_Encoder(in_dim, hs, droprate)
        self.decoder = DSR_Decoder(hs//2, hs, hs, droprate)
        self.sem_cls = ClassClassifier(hs, 2, droprate)
        self.sem_dom = DomainClassifier(hs)
        self.dom_cls = ClassClassifier(hs, 2, droprate)
        self.dom_dom = DomainClassifier(hs)

    def forward(self, x, vts, adj, domain, grl_lamda=0.1, recon=True):
        if domain == 0:
            x = torch.cat((x, self.semb(vts)), dim=-1)
            x = torch.cat((x, self.svf(vts)), dim=-1)
        else:
            x = torch.cat((x, self.temb(vts)), dim=-1)
            x = torch.cat((x, self.tvf(vts)), dim=-1)

        x = F.instance_norm(x)
        res = self.encoder(x, adj)
        if recon:
            res['a_recons'] = self.decoder(res['d'], res['y'])
            res['sem_dom'] = self.sem_dom(grad_reverse(res['y'], grl_lamda))
            res['dom_cls'] = self.dom_cls(grad_reverse(res['d'], grl_lamda))
            res['dom_dom'] = self.dom_dom(res['d'])

        res['sem_cls'] = self.sem_cls(res['y'])

        return res





