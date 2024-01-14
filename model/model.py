import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .modules import Tacotron2Encoder, ContentEncoder, Decoder, Aligner, Predictor
from .attention import BidirectionalAttention, BidirectionalAdditiveAttention, CTCAttention
from .position import PositionalEncoding


class FAModel(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.text_encoder = Tacotron2Encoder(hparams.text_encoder)
        self.speech_encoder = ContentEncoder(hparams.speech_encoder)
        self.attention = (BidirectionalAttention if hparams.model == 'NeuFA' else CTCAttention)(hparams.attention)
        self.text_decoder = Decoder(hparams.text_decoder)
        self.speech_decoder = Decoder(hparams.speech_decoder)
        self.positional_encoding_text = PositionalEncoding(hparams.text_encoder.output_dim)
        self.positional_encoding_speech = PositionalEncoding(hparams.speech_encoder.output_dim)
        self.aligner = Aligner(hparams.aligner)
        #self.aligner = Predictor(hparams.predictor)
        self.hparams = hparams
        if hparams.Tep:
            self.tep = nn.Linear(hparams.speech_encoder.output_dim, 1)
        if hparams.MeP:
            self.mep = nn.Linear(hparams.text_encoder.output_dim, 1)

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.cross_entrophy = torch.nn.CrossEntropyLoss()

    def encode(self, texts, mfccs):
        text_lengths = [i.shape[0] for i in texts]
        texts = pad_sequence(texts, batch_first=True)
        texts = self.text_encoder(texts, text_lengths)
        texts = torch.cat([texts, torch.zeros((texts.shape[0], 1, texts.shape[2]), device=texts.device)], axis=-2)

        mfcc_lengths = [i.shape[0] for i in mfccs]
        mfccs = pad_sequence(mfccs, batch_first=True)
        mfccs = self.speech_encoder(mfccs, mfcc_lengths)
        mfccs = torch.cat([mfccs, torch.zeros((mfccs.shape[0], 1, mfccs.shape[2]), device=mfccs.device)], axis=-2)

        return texts, mfccs, text_lengths, mfcc_lengths

    def positional_encoding(self, texts, mfccs, text_lengths=None, mfcc_lengths=None, calc_loss=False):
        optional = {}

        if self.hparams.TeP:
            tep = torch.relu(self.tep(mfccs)).squeeze(-1)
            tep = torch.cumsum(tep, dim=-1)
            if calc_loss:
                p_text_lengths = [tep[i][l-1] for i, l in enumerate(mfcc_lengths)]
                optional['tep_loss'] = self.length_loss(text_lengths, p_text_lengths)
                with torch.no_grad():
                    optional['tep_mse'] = self.length_loss(text_lengths, p_text_lengths, normalize=False)

            mfccs_pe = self.positional_encoding_speech(mfccs, tep)
            if self.hparams.double_PE:
                mfccs_pe = torch.cat([mfccs_pe, self.positional_encoding_speech(mfccs)], dim=-1)
        else:
            mfccs_pe = self.positional_encoding_speech(mfccs)

        if self.hparams.MeP:
            mep = 10 * torch.relu(self.mep(texts)).squeeze(-1)
            mep = torch.cumsum(mep, dim=-1)
            if calc_loss:
                p_mfcc_lengths = [mep[i][l-1] for i, l in enumerate(text_lengths)]
                optional['mep_loss'] = self.length_loss(mfcc_lengths, p_mfcc_lengths)
                with torch.no_grad():
                    optional['mep_mse'] = self.length_loss(mfcc_lengths, p_mfcc_lengths, normalize=False)

            texts_pe = self.positional_encoding_text(texts, mep)
            if self.hparams.double_PE:
                texts_pe = torch.cat([texts_pe, self.positional_encoding_text(texts)], dim=-1)
        else:
            texts_pe = self.positional_encoding_text(texts)

        return texts_pe, mfccs_pe, optional

    def length_loss(self, lengths, p_lengths, normalize=True):
        p_lengths = torch.stack(p_lengths)
        lengths = torch.FloatTensor(lengths).to(p_lengths.device)
        if normalize:
            p_lengths = p_lengths / lengths.detach()
            lengths = lengths / lengths.detach()
            return self.mae(lengths, p_lengths)
        else:
            return self.mse(lengths, p_lengths)

    def decode(self, texts_at_frame, mfccs_at_text, text_lengths, mfcc_lengths):
        p_texts = self.text_decoder(mfccs_at_text, text_lengths)
        p_mfccs = self.speech_decoder(texts_at_frame, mfcc_lengths)

        return p_texts, p_mfccs

    def forward(self, texts, mfccs, calc_loss=False, boundary_threshold=None, boundary_gt=None):
        # If calc_loss is specified, tep_loss, tep_mse, mep_loss, mep_mse are calculated where applicable.
        # attention_loss, boundary_loss, boundary_mae should be managed from the outside.
        # If boundary_threshold is None, don't calculating anything related to boundary
        texts, mfccs, text_lengths, mfcc_lengths = self.encode(texts, mfccs)
        texts_pe, mfccs_pe, optional = self.positional_encoding(texts, mfccs, text_lengths, mfcc_lengths, calc_loss)

        texts_at_frame, mfccs_at_text, w, *ctc_ret = self.attention(texts_pe, mfccs_pe, texts, mfccs, text_lengths, mfcc_lengths)
        if ctc_ret:
            logprobs, optional['ctc_loss'] = ctc_ret

        if boundary_threshold:
            if self.hparams.aligner.enable:
                p_boundaries = self.aligner(texts[:,:-1,:], w, text_lengths, mfcc_lengths) 
                boundary_loss = self.boundary_loss(p_boundaries, boundary_gt)
                boundaries = self.extract_boundary(p_boundaries, boundary_threshold)
            else:
                if self.hparams.model != 'CAFA':
                    raise ValueError
                boundary_loss = self.attention.ctc.supervised_loss(logprobs, boundary_gt)
                boundaries = self.attention.ctc.extract(logprobs, text_lengths, mfcc_lengths).to(torch.float) / 100
            optional['boundary_loss'] = boundary_loss
            optional['boundaries'] = boundaries

        p_texts, p_mfccs = self.decode(texts_at_frame, mfccs_at_text, text_lengths, mfcc_lengths)

        w = [x[:l1, :l2] for x, l1, l2 in zip(w, text_lengths, mfcc_lengths)]
        return p_texts, p_mfccs, w, optional

    def text_loss(self, p_texts, texts):
        p_texts = torch.cat(p_texts)
        texts = torch.cat(texts)
        return self.cross_entrophy(p_texts, texts)

    def mfcc_loss(self, p_mfccs, mfccs):
        p_mfccs = torch.cat(p_mfccs)
        mfccs = torch.cat(mfccs)
        return self.mse(p_mfccs, mfccs)

    def attention_loss(self, w, alpha=0.5):
        loss = []
        for x in w:
            x = torch.amax(x, dim=-1)
            a = torch.linspace(1e-6, 1, w.shape[0], device=w.device).repeat(w.shape[1], 1).T
            b = torch.linspace(1e-6, 1, w.shape[1], device=w.device).repeat(w.shape[0], 1)
            r1 = torch.maximum((a / b), (b / a))
            r2 = torch.maximum(a.flip(1) / b.flip(0), b.flip(0)/ a.flip(1))
            r = torch.maximum(r1, r2) - 1
            r = torch.tanh(alpha * r)
            loss.append(torch.mean(w * r.detach()))
        loss = torch.stack(loss)
        return torch.mean(loss)

    def extract_boundary(self, p_boundaries, threshold=0.5):
        result = []
        for p_boundary in p_boundaries:
            result.append([])
            result[-1].append(torch.FloatTensor([i[i<threshold].shape[0] / 100 for i in p_boundary[:,0,:]]))
            result[-1].append(torch.FloatTensor([i[i<threshold].shape[0] / 100 for i in p_boundary[:,1,:]]))
            result[-1] = torch.stack(result[-1], dim=-1).to(p_boundaries[0].device)
        return result

    def boundary_loss(self, p_boundaries, boundaries):
        boundaries = [i.reshape((-1, 1)) for i in boundaries]
        p_boundaries = [i.reshape((-1, 1, i.shape[2])) for i in p_boundaries]
        p_boundaries = [p_boundary[boundaries[i]>-1] for i, p_boundary in enumerate(p_boundaries)]
        boundaries = [i[i>-1] for i in boundaries]
        gated_boundaries = [torch.zeros(i.shape, device=p_boundaries[0].device) for i in p_boundaries]
        for i, boundary in enumerate(boundaries):
            for j, b in enumerate(boundary):
                #if j == 0:
                    gated_boundaries[i][j, int(100 * b):] = 1
                #else:
                #    gated_boundaries[i][j, :int(100 * b)] = 1
        boundaries = [i.reshape((-1,1)) for i in gated_boundaries]
        p_boundaries = [i.reshape((-1,1)) for i in p_boundaries]
        boundaries = torch.cat(boundaries)
        p_boundaries = torch.cat(p_boundaries)
        return self.mae(p_boundaries, boundaries)

    def boundary_mae(self, p_boundaries, boundaries):
        boundaries = [i.reshape((-1,1)) for i in boundaries]
        p_boundaries = [i.reshape((-1,1)) for i in p_boundaries]
        p_boundaries = [p_boundary[boundaries[i]>-1] for i, p_boundary in enumerate(p_boundaries)]
        boundaries = [i[i>-1] for i in boundaries]
        boundaries = torch.cat(boundaries)
        p_boundaries = torch.cat(p_boundaries)
        #print(torch.median(torch.abs(p_boundaries - boundaries)))
        return self.mae(p_boundaries, boundaries)


if __name__ == '__main__':
    exit(0)
    import os
    from data.buckeye import Buckeye
    from data.common import Collate
    from hparams import base, temp

    device = 'cuda:5'
    batch_size = 4

    dataset = Buckeye(os.path.expanduser('~/BuckeyeTrain'), reduction=base.reduction_rate)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Collate(device), drop_last=True)

    for batch in data_loader:
        for Model in [NeuFA_TeMP, NeuFA_MeP, NeuFA_TeP, NeuFA_base]:
            if Model == NeuFA_TeMP:
                model = Model(temp)
            else:
                model = Model(base)
            model.to(device)
            output = model(*batch[:2])
            #print([i.shape for i in output])
            print(model.text_loss(output[0], batch[0]))
            print(model.mfcc_loss(output[1], batch[1]))
            print(model.boundary_loss(output[-1], batch[2]))
            print(model.boundary_mae(model.extract_boundary(output[-1]), batch[2]))
            print(model.attention_loss(output[2], output[3]))
            if Model in [NeuFA_TeP, NeuFA_MeP, NeuFA_TeMP]:
                print(model.length_loss(output[4], output[5]))
            if Model in [NeuFA_TeMP]:
                print(model.length_loss(output[6], output[7]))
            break
        break
