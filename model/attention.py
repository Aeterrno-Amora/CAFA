import torch
from torch import nn
from torch.nn import functional as F

NINF = -float('inf')

class BidirectionalAttention(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.k1_layer = nn.Linear(hparams.text_input_dim, hparams.dim)
        self.k2_layer = nn.Linear(hparams.speech_input_dim, hparams.dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, k1, k2, v1, v2, k1_lengths, k2_lengths):
        k1 = self.k1_layer(k1)
        k2 = self.k2_layer(k2)
        score = torch.bmm(k1, k2.mT)

        mask = torch.zeros_like(score, dtype=torch.bool).detach()
        for i, l in enumerate(k1_lengths):
            mask[i,l:,:].logical_not_()
        for i, l in enumerate(k2_lengths):
            mask[i,:,l:].logical_not_()
        score = score.masked_fill_(mask, NINF)

        w1 = self.softmax(score.mT)
        w2 = self.softmax(score)

        o1 = torch.bmm(w1, v1)
        o2 = torch.bmm(w2, v2)

        w = torch.stack([w1.mT, w2], dim=-1)
        # w = [torch.stack((x1[:l2, :l1].mT, x2[:l1, :l2]), dim=-1) for x1, x2, l1, l2 in zip(w1, w2, k1_lengths, k2_lengths)]
        # score = [i[:l1, :l2] for i, l1, l2 in zip(score, k1_lengths, k2_lengths)]

        return o1, o2, w

class BidirectionalAdditiveAttention(nn.Module):
    # deprecated, may not be compatible

    def __init__(self, k1_dim, k2_dim, attention_dim):
        super().__init__()
        self.k1_layer = nn.Linear(k1_dim, attention_dim)
        self.k2_layer = nn.Linear(k2_dim, attention_dim)
        self.score_layer = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

    def forward(self, k1, k2, v1, v2, k1_lengths=None, k2_lengths=None):
        k1 = self.k1_layer(k1).repeat(k2.shape[1], 1, 1, 1).permute(1,2,0,3)
        k2 = self.k2_layer(k2).repeat(k1.shape[1], 1, 1, 1).permute(1,0,2,3)
        score = self.score_layer(self.tanh(k1 + k2)).squeeze(-1)

        if k1_lengths or k2_lengths:
            mask = torch.zeros(score.shape, dtype=torch.int32).detach().to(score.device)
            for i, l in enumerate(k1_lengths):
                mask[i,l:,:] += 1
            for i, l in enumerate(k2_lengths):
                mask[i,:,l:] += 1
            mask = mask == 1
            score = score.masked_fill_(mask, NINF)

        w1 = self.softmax1(score.mT)
        w2 = self.softmax2(score)

        o1 = torch.bmm(w1, v1)
        o2 = torch.bmm(w2, v2)

        w1 = [i[:l2, :l1] for i, l1, l2 in zip(w1, k1_lengths, k2_lengths)]
        w2 = [i[:l1, :l2] for i, l1, l2 in zip(w2, k1_lengths, k2_lengths)]
        score = [i[:l1, :l2] for i, l1, l2 in zip(score, k1_lengths, k2_lengths)]

        return o1, o2, w1, w2, score


class GeneralizedCTC(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.pad_silences = hparams.pad_silences
        self.fast_sliding_window = hparams.fast_sliding_window

    def _dp_noskip(self, forward): # please pad logprobs by 1, and this is done inplace in logprobs
        # forward[j, i] = logsumexp(forward[j-1, i-1], forward[j, i-1]) + logprobs[j, i]
        forward[:, 2:, 0] = NINF
        if self.fast_sliding_window:
            sliding_window = forward.unfold(-2, 2, 1) # j-1 (full), j (full)
            for i in range(1, forward.size(-1)):
                forward[:, 2:, i] += torch.logsumexp(sliding_window[:, :, i-1, :], dim=-1)
        else:
            for i in range(1, forward.size(-1)):
                forward[:, 2:, i] += torch.logaddexp(forward[:, 1:-1, i-1], forward[:, 2:, i-1])

    def _dp_skip(self, forward): # please pad logprobs by 2, and this is done inplace in logprobs
        # forward[j, i] = logsumexp((forward[j-2, i-1] if j%2==0), forward[j-1, i-1], forward[j, i-1]) + logprobs[j, i]
        odd_mask = (torch.arange(forward.size(-3) - 2, device=forward.device) % 2 != 0)[None, :, None]
        forward[:, 4:, 0] = NINF
        if self.fast_sliding_window:
            sliding_window = forward.unfold(-2, 3, 1) # j-2 (odd), j-1 (full), j (full)
            mask = torch.stack([odd_mask] + [torch.ones_like(odd_mask)] * 2, dim=-1)
            for i in range(1, forward.size(-1)):
                forward[:, 2:, i] += torch.logsumexp(torch.where(mask, sliding_window[:, :, i-1, :], NINF), dim=-1)
        else:
            for i in range(1, forward.size(-1)):
                forward[:, 2:, i] += torch.logaddexp(torch.where(odd_mask, forward[:, :-2, i-1], NINF),
                                        torch.logaddexp(forward[:, 1:-1, i-1], forward[:, 2:, i-1]))

        # input_len_tensor = torch.tensor(input_lengths, dtype=torch.int32, )
        # backward = F.pad(logprobs, (0, 0, 0, 2), value=NINF)
        # sliding_window = backward.unfold(-2, 3, 1)
        # odd_mask = (torch.arange(sliding_window.size(-3)) % 2 != 0)[None, :, None]
        # mask = torch.stack([torch.ones_like(odd_mask)] * 2 + [odd_mask], dim=-1) # j (full), j+1 (full), j+2 (odd)
        # active = torch.zeros((backward.size(0),), dtype=torch.bool)
        # for x, l1, l2 in zip(backward, target_lengths, input_lengths):
        #     x[:l1-2, l2-1] = NINF
        # for i in range(max_in_len-2, -1, -1):
        #     val, idx = torch.logsumexp(torch.where(mask, sliding_window[:, :, i, :], NINF), dim=-1)
        #     backward[:, :-2, i] += val
        #     decision[:, :, i] = idx

    def forward(self, logprobs, target_lengths, input_lengths):
        # logprobs [N, max(target_lengths*2+1), max(input_lengths)] obtained from log_softmax(dim=-2)
        # *2+1 is the silence padding in every even indices
        # shape [k, target_lengths[k]*2+1, input_lengths[k]]

        pad = 2 if self.pad_silences else 1
        forward = F.pad(logprobs, (0, 0, pad, 0), value=NINF)
        backward = forward.clone()
        for x, l1, l2 in zip(backward, target_lengths, input_lengths):
            x[pad : pad+l1, :l2] = x[pad : pad+l1, :l2].flip(-1, -2)

        if self.pad_silences:
            target_lengths = [l*2+1 for l in target_lengths]
            self._dp_skip(forward)
            self._dp_skip(backward)
        else:
            self._dp_noskip(forward)
            self._dp_noskip(backward)

        for x, l1, l2 in zip(backward, target_lengths, input_lengths):
            x[pad : pad+l1, :l2] = x[pad : pad+l1, :l2].flip(-1, -2)

        alignment = forward[:, pad:, :] + backward[:, pad:, :] - logprobs
        if self.pad_silences:
            ctc_loss = torch.logaddexp(backward[:, 0, 0], backward[:, 1, 0])
        else:
            ctc_loss = backward[:, 0, 0]

        return alignment.nan_to_num_(NINF, None, NINF), ctc_loss

    def extract(self, logprobs, target_lengths, input_lengths):
        with torch.no_grad():
            # extract the most probable hard alignment
            n = len(input_lengths)

            pad = 2 if self.pad_silences else 1
            forward = F.pad(logprobs, (0, 0, pad, 0), value=NINF)
            decision = torch.empty_like(logprobs, dtype=torch.int32)
            decision[:, :, 0] = 0
            if self.pad_silences:
                odd_mask = (torch.arange(forward.size(-3) - 2, device=forward.device) % 2 != 0)[None, :, None]
                forward[:, 4:, 0] = NINF
                sliding_window = forward.unfold(-2, 3, 1) # j-2 (odd), j-1 (full), j (full)
                mask = torch.stack([odd_mask] + [torch.ones_like(odd_mask)] * 2, dim=-1)
                for i in range(1, forward.size(-1)):
                    val, idx = torch.max(torch.where(mask, sliding_window[:, :, i-1, :], NINF), dim=-1)
                    forward[:, 2:, i] += val
                    decision[:, :, i] = idx
                decision = 2 - decision
            else:
                forward[:, 2:, 0] = NINF
                sliding_window = forward.unfold(-2, 2, 1) # j-1 (full), j (full)
                for i in range(1, forward.size(-1)):
                    val, idx = torch.logsumexp(sliding_window[:, :, i-1, :], dim=-1)
                    forward[:, 2:, i] += val
                    decision[:, :, i] = idx
                decision = 1 - decision

            input_len_tensor = torch.tensor(input_lengths, dtype=torch.int32, device=logprobs.device)
            target_len_tensor = torch.tensor(target_lengths, dtype=torch.int32, device=logprobs.device)
            if self.pad_silences:
                target_len_tensor = target_len_tensor * 2 + 1
            input_to_target = torch.full((n, logprobs.size(-1)), -1, dtype=torch.int32, device=logprobs.device)
            input_to_target[range(n), input_len_tensor-1] = target_len_tensor-1
            for i in range(forward.size(-1)-1, 0):
                j = input_to_target[:, i]
                ids = (j >= 0).nonzero(as_tuple=True)[0]
                d = decision[ids, j[ids], i]
                input_to_target[ids, i-1] = input_to_target[ids, i] - d
            idx = torch.arange(logprobs.size(-2), dtype=torch.int32, device=logprobs.device)[None, :]
            target_range = torch.stack([
                torch.searchsorted(input_to_target, idx, side='left', out_int32=True),
                torch.searchsorted(input_to_target, idx, side='right', out_int32=True),
            ], dim=-1)
            if self.pad_silences:
                target_range = target_range[:, 1::2]
            return target_range

    def supervised_loss(self, logprobs, target_range_gt):
        target_range_gt = (target_range_gt * 100).to(torch.int32) # dataset specific scaling, determined by hop size
        boundaries = torch.flatten(target_range_gt, start_dim=1) if self.pad_silences else target_range_gt[:, 1:, 0]
        input_to_target = torch.searchsorted(boundaries.to(logprobs.device),
                    torch.arange(logprobs.size(-1), dtype=torch.int32, device=logprobs.device)[None, :], out_int32=True)
        return F.nll_loss(logprobs, input_to_target)

# def renorm(x):
#     return x / torch.max(torch.norm(x, p=1, dim=-1, keepdim=True), torch.tensor(1e-4))

class CTCAttention(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.silence_k = nn.Parameter(torch.randn(hparams.dim)) # TODO: init?
        self.silence_v = nn.Parameter(torch.randn(hparams.text_input_dim)) # v1_dim # TODO: init?
        self.target_layer = nn.Linear(hparams.text_input_dim, hparams.dim)
        self.input_layer = nn.Linear(hparams.speech_input_dim, hparams.dim)
        self.ctc = GeneralizedCTC(hparams)
        self.pad_silences = hparams.pad_silences and hparams.same_silence
        if hparams.pad_silences and not hparams.same_silence:
            raise NotImplementedError('Need to pad sequences before text encoder, and review the lengths.')

    def encode(self, target, input, target_lengths, input_lengths):
        target = self.target_layer(target)
        input = self.input_layer(input)

        if self.pad_silences: # pad target with a silence token
            target = torch.cat([self.silence_k[None, None, :], target])

        logits = torch.bmm(target, input.mT)

        if self.pad_silences: # expand silence target[0] to every even indices
            old = logits
            logits = old[:, 0, :].repeat(old.size(-2) * 2 - 1, 1) # silence
            logits[:, 1::2, :] = old[:, 1:, :] # non-silences

        for x, l in zip(logits, target_lengths):
            x[self.pad_silences + l :, :] = NINF
        logprobs = F.log_softmax(logits, dim=-2) # calculate probs after repeating tokens
        for x, l in zip(logprobs, input_lengths):
            x[:, l:] = NINF

        return logprobs

    def forward(self, target, input, v1, v2, target_lengths, input_lengths):
        # v1, v2 are v_target, v_input respectively

        logprobs = self.encode(target, input, target_lengths, input_lengths)

        alignment, ctc_loss = self.ctc(logprobs, target_lengths, input_lengths)

        w1 = (alignment - ctc_loss).exp().nan_to_num_(0, 0, 0)
        w2 = alignment.softmax(dim=-1).nan_to_num_(0, 0, 0)
        if self.pad_silences:
            target = torch.cat([self.silence_k[None, None, :], target])
            o1 = torch.bmm(w1[:, 1::2, :].mT, v1) + \
                 torch.bmm(w1[:, 0::2, :].sum(dim=-2, keepdim=True).mT, self.silence_v)
            o2 = torch.bmm(w2[:, 1::2, :], v2)
        else:
            o1 = torch.bmm(w1.mT, v1)
            o2 = torch.bmm(w2, v2)

        # alignment = [i[:l1, :l2] for i, l1, l2 in zip(alignment, target_lengths, input_lengths)]
        return o1, o2, w1.unsqueeze(-1), logprobs, ctc_loss
