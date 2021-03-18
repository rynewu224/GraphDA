import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd.function import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class BatchGraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BatchGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        expand_weight = self.weight.expand(x.shape[0], -1, -1)
        output = torch.bmm(adj, torch.bmm(x, expand_weight))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, in_features, out_features, attn_dropout):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.in_features = in_features
        self.out_features = out_features
        self.w = Parameter(torch.Tensor(n_head, in_features, out_features))
        self.a_src = Parameter(torch.Tensor(n_head, out_features, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, out_features, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        self.bias = Parameter(torch.Tensor(out_features))
        init.constant_(self.bias, 0)
        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj):
        bs, n = x.size()[:2]  # x = (bs, n, in_dim)
        h_prime = torch.matmul(x.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(F.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(F.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3,
                                                                                       2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = ~adj.unsqueeze(1)  # bs x 1 x n x n
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        output += self.bias
        output = output.view(bs, n, -1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BatchGIN(Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(BatchGIN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin_0 = nn.Linear(in_features, hidden_size)
        self.lin_1 = nn.Linear(hidden_size, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        h = x + torch.bmm(adj, x)
        h = self.dropout(self.act(self.lin_0(h)))
        h = self.lin_1(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)


class BatchInnerProductDecoder(nn.Module):
    def __init__(self):
        super(BatchInnerProductDecoder, self).__init__()

    def forward(self, z):
        return torch.bmm(z, z.permute(0, 2, 1))

