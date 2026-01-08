# model_gatv2.py
import torch
import torch.nn as nn

class GATv2Conv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, feat_drop=0.0, attn_drop=0.0,
                 negative_slope=0.2, residual=True, concat=True, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        self.concat = concat

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels * heads if concat else out_channels))
        else:
            self.register_parameter("bias", None)

        if residual:
            out_dim = out_channels * heads if concat else out_channels
            if in_channels != out_dim:
                self.res_fc = nn.Linear(in_channels, out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_uniform_(self.res_fc.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def _edge_softmax(self, dst, e, num_nodes):
        H = e.size(1)
        minus_inf = torch.full((num_nodes, H), -float("inf"), device=e.device, dtype=e.dtype)
        max_per_dst = torch.scatter_reduce(
            minus_inf, 0,
            dst.unsqueeze(-1).expand(-1, H),
            e, reduce="amax", include_self=True
        )
        m = max_per_dst.index_select(0, dst)
        e_exp = torch.exp(e - m)
        sum_per_dst = torch.zeros((num_nodes, H), device=e.device, dtype=e.dtype)
        sum_per_dst.index_add_(0, dst, e_exp)
        denom = sum_per_dst.index_select(0, dst)
        return e_exp / (denom + 1e-16)

    def forward(self, x, edge_index, num_nodes=None):
        N = x.size(0) if num_nodes is None else num_nodes
        src, dst = edge_index[0], edge_index[1]

        h = self.lin(x).view(N, self.heads, self.out_channels)
        h = self.feat_drop(h)

        h_src = h.index_select(0, src)
        h_dst = h.index_select(0, dst)

        e_ij = self.leaky_relu(h_src + h_dst)
        e_ij = (e_ij * self.att).sum(dim=-1)

        alpha = self._edge_softmax(dst, e_ij, N)
        alpha = self.attn_drop(alpha)
        m_ij = h_src * alpha.unsqueeze(-1)

        out = torch.zeros((N, self.heads, self.out_channels), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, m_ij)

        out = out.reshape(N, self.heads * self.out_channels) if self.concat else out.mean(dim=1)
        if self.res_fc is not None:
            out = out + self.res_fc(x)
        if self.bias is not None:
            out = out + self.bias
        return out


class GATv2Net(nn.Module):
    def __init__(self, in_dim, num_hidden, num_classes, num_layers, heads,
                 activation=nn.ReLU(), feat_drop=0.0, attn_drop=0.0,
                 negative_slope=0.2, residual=True):
        super().__init__()
        assert len(heads) == num_layers + 1
        self.layers = nn.ModuleList()
        self.activ = activation

        self.layers.append(GATv2Conv(in_dim, num_hidden, heads=heads[0],
                                     feat_drop=feat_drop, attn_drop=attn_drop,
                                     negative_slope=negative_slope, residual=residual, concat=True))
        for l in range(1, num_layers):
            in_ch = num_hidden * heads[l-1]
            self.layers.append(GATv2Conv(in_ch, num_hidden, heads=heads[l],
                                         feat_drop=feat_drop, attn_drop=attn_drop,
                                         negative_slope=negative_slope, residual=residual, concat=True))
        in_ch = num_hidden * heads[num_layers-1] if num_layers > 0 else num_hidden * heads[0]
        self.layers.append(GATv2Conv(in_ch, num_classes, heads=heads[-1],
                                     feat_drop=feat_drop, attn_drop=attn_drop,
                                     negative_slope=negative_slope, residual=False, concat=False))

    def forward(self, x, edge_index, num_nodes=None):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, num_nodes)
            if self.activ is not None:
                x = self.activ(x)
        x = self.layers[-1](x, edge_index, num_nodes)
        return x
