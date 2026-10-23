import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1):

    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    attn = torch.bmm(contextT, query) 
    attn = attn.view(batch_size*sourceL, queryL)
    attn = torch.softmax(attn, dim=-1)  
    attn = attn.view(batch_size, sourceL, queryL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    attn = attn * gamma1
    attn = torch.softmax(attn, dim=-1)
    attn = attn.view(batch_size, queryL, sourceL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class MultiSpatialAttention(nn.Module):
    def __init__(self, idf, cdf, num_head):
        super(MultiSpatialAttention, self).__init__()
        self.num_head = num_head
        # ✅ register parameters
        self.conv_context = nn.ModuleList([conv1x1(cdf, idf) for _ in range(num_head)])
        self.sm = nn.Softmax(dim=-1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        target = input.view(batch_size, -1, queryL)            # B x idf x Q
        targetT = torch.transpose(target, 1, 2).contiguous()   # B x Q x idf
        sourceIn = context.unsqueeze(3)                        # B x cdf x S x 1

        attnOut, weightedContextOut = None, None
        for i in range(self.num_head):
            sourceT = self.conv_context[i](sourceIn).squeeze(3)   # B x idf x S
            attn = torch.bmm(targetT, sourceT)                    # B x Q x S
            attn = attn.view(batch_size * queryL, sourceL)

            if self.mask is not None:
                mask = self.mask.repeat(queryL, 1)
                attn.data.masked_fill_(mask.data, -float('inf'))

            attn = self.sm(attn)
            attn = attn.view(batch_size, queryL, sourceL)
            attn = torch.transpose(attn, 1, 2).contiguous()       # B x S x Q

            weightedContext = torch.bmm(sourceT, attn)             # B x idf x Q
            attn_map = attn.view(batch_size, -1, ih, iw)           # B x S x ih x iw

            if i == 0:
                weightedContextOut = weightedContext
                attnOut = attn_map
            else:
                weightedContextOut = weightedContextOut + weightedContext
                attnOut = attnOut + attn_map

        return weightedContextOut, attnOut


class MultiChannelAttention(nn.Module):
    def __init__(self, idf, cdf, size, num_head):
        super(MultiChannelAttention, self).__init__()
        self.num_head = num_head
        # ✅ register parameters
        self.conv_context = nn.ModuleList([conv1x1(cdf, size * size) for _ in range(num_head)])
        self.sm = nn.Softmax(dim=-1)
        self.idf = idf

    def forward(self, weightedContext, context, ih, iw):
        batch_size, sourceL = context.size(0), context.size(2)
        sourceIn = context.unsqueeze(3)  # B x cdf x S x 1

        attnOut, weightedContextOut = None, None
        for i in range(self.num_head):
            sourceC = self.conv_context[i](sourceIn).squeeze(3)   # B x (size*size) x S

            attn_c = torch.bmm(weightedContext, sourceC)          # B x idf x S
            attn_c = attn_c.view(batch_size * self.idf, sourceL)
            attn_c = self.sm(attn_c)
            attn_c = attn_c.view(batch_size, self.idf, sourceL)
            attn_c = torch.transpose(attn_c, 1, 2).contiguous()   # B x S x idf

            weightedContext_T = torch.bmm(sourceC, attn_c)         # B x (size*size) x idf
            weightedContext_T = torch.transpose(weightedContext_T, 1, 2).contiguous()
            weightedContext_T = weightedContext_T.view(batch_size, -1, ih, iw)

            if i == 0:
                weightedContextOut = weightedContext_T
                attnOut = attn_c
            else:
                weightedContextOut = weightedContextOut + weightedContext_T
                attnOut = attnOut + attn_c

        return weightedContextOut, attnOut
