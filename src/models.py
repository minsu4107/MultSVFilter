import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder

class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_v, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_v, hyp_params.orig_d_l
        self.d_l, self.d_a, self.d_v = 10,10,10
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 510      # assuming d_l == d_a == d_v
        else:
            combined_dim = 510 * 3
        project_1_out = int(combined_dim / 3)
        project_2_out = int(project_1_out / 3)
        
        project_1_dim = [combined_dim, project_1_out]
        project_2_dim = [project_1_out, project_2_out]

        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=10, padding='same', padding_mode='replicate', bias=True)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=10, padding='same', padding_mode='replicate', bias=True)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=10, padding='same', padding_mode='replicate', bias=True)
        
        self.proj_l_1 = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=10, padding='same', padding_mode='replicate', bias=True)
        self.proj_a_1 = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=10, padding='same', padding_mode='replicate', bias=True)
        self.proj_v_1 = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=10, padding='same', padding_mode='replicate', bias=True)
        
        self.proj_l_2 = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=10, padding='same', padding_mode='replicate', bias=True)
        self.proj_a_2 = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=10, padding='same', padding_mode='replicate', bias=True)
        self.proj_v_2 = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=10, padding='same', padding_mode='replicate', bias=True)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a_1 = self.get_network(self_type='la')
            self.trans_l_with_v_1 = self.get_network(self_type='lv')
            
            self.trans_l_with_a_2 = self.get_network(self_type='la')
            self.trans_l_with_v_2 = self.get_network(self_type='lv')
            
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
            
        if self.aonly:
            self.trans_a_with_l_1 = self.get_network(self_type='al')
            self.trans_a_with_v_1 = self.get_network(self_type='av')
            
            self.trans_a_with_l_2 = self.get_network(self_type='al')
            self.trans_a_with_v_2 = self.get_network(self_type='av')
            
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
            
        if self.vonly:
            self.trans_v_with_l_1 = self.get_network(self_type='vl')
            self.trans_v_with_a_1 = self.get_network(self_type='va')
            
            self.trans_v_with_l_2 = self.get_network(self_type='vl')
            self.trans_v_with_a_2 = self.get_network(self_type='va')
            
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem_1 = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem_1 = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem_1 = self.get_network(self_type='v_mem', layers=3)
        
        self.trans_l_mem_2 = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem_2 = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem_2 = self.get_network(self_type='v_mem', layers=3)
        
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.adaptive2D_l = nn.AdaptiveAvgPool2d((None, 1))
        self.adaptive2D_a = nn.AdaptiveAvgPool2d((None, 1))
        self.adaptive2D_v = nn.AdaptiveAvgPool2d((None, 1))
        
        self.adaptive2D_l_1 = nn.AdaptiveAvgPool2d((None, 1))
        self.adaptive2D_a_1 = nn.AdaptiveAvgPool2d((None, 1))
        self.adaptive2D_v_1 = nn.AdaptiveAvgPool2d((None, 1))
        
        self.adaptive2D_l_2 = nn.AdaptiveAvgPool2d((None, 1))
        self.adaptive2D_a_2 = nn.AdaptiveAvgPool2d((None, 1))
        self.adaptive2D_v_2 = nn.AdaptiveAvgPool2d((None, 1))
        
        self.proj1 = nn.Linear(project_1_dim[0], project_1_dim[1])
        self.proj2 = nn.Linear(project_2_dim[0] , project_2_dim[1])

        self.out_layer = nn.Linear(project_2_dim[1], output_dim)
        
        self.activation = torch.nn.Sigmoid()

    def get_network(self, self_type='l', layers=-1):
        position_fg = False
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
            position_fg = True
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
            position_fg = True
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
            position_fg = True
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
            position_fg = True
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
            position_fg = True
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
            position_fg = True
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  position_fg=position_fg)

    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        # left breakpoint
        x_l_1 = F.dropout(x_l[:,:255].transpose(1, 2), p=self.embed_dropout, training=self.training).float()
        x_a_1 = x_a[:,255:].transpose(1, 2).float()
        x_v_1 = x_v[:,255:].transpose(1, 2).float()
        
        # right breakpoint
        x_l_2 = F.dropout(x_l[:,:255].transpose(1, 2), p=self.embed_dropout, training=self.training).float()
        x_a_2 = x_a[:,255:].transpose(1, 2).float()
        x_v_2 = x_v[:,255:].transpose(1, 2).float()
        

        # left breakpoint
        # Project the textual/visual/audio features
        
        proj_x_l = x_l_1 if self.orig_d_l == self.d_l else self.proj_l_1(x_l_1)
        proj_x_a = x_a_1 if self.orig_d_a == self.d_a else self.proj_a_1(x_a_1)
        proj_x_v = x_v_1 if self.orig_d_v == self.d_v else self.proj_v_1(x_v_1)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a_1(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v_1(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem_1(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l_1 = last_hs_1 = self.adaptive2D_l_1(h_ls)

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l_1(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v_1(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem_1(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a_1 = last_hs_1 = self.adaptive2D_a_1(h_as)

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l_1(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a_1(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem_1(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v_1 = last_hs_1 = self.adaptive2D_v_1(h_vs)

        # right breakpoint
        # Project the textual/visual/audio features
        proj_x_l = x_l_2 if self.orig_d_l == self.d_l else self.proj_l_2(x_l_2)
        proj_x_a = x_a_2 if self.orig_d_a == self.d_a else self.proj_a_2(x_a_2)
        proj_x_v = x_v_2 if self.orig_d_v == self.d_v else self.proj_v_2(x_v_2)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a_2(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v_2(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem_2(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l_2 = last_hs_2 = self.adaptive2D_l_2(h_ls)

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l_2(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v_2(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem_2(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a_2 = last_hs_2 = self.adaptive2D_a_2(h_as)

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l_2(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a_2(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem_2(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v_2 = last_hs_2 = self.adaptive2D_v_2(h_vs)

        if self.partial_mode == 3:
            last_hs_1 = torch.cat([last_h_l_1, last_h_a_1, last_h_v_1], dim=0)
            last_hs_2 = torch.cat([last_h_l_2, last_h_a_2, last_h_v_2], dim=0)
            last_hs = torch.cat([last_hs_1, last_hs_2], dim=0)
        else:
            last_hs = torch.cat([last_hs_1, last_hs_2], dim=0)

        last_hs = last_hs.transpose(2,0)
        last_hs = last_hs.squeeze(dim=0)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        output = self.out_layer(last_hs_proj)
        act_output = self.activation(output)
        
        return act_output.reshape(-1), output