import onmt
import torch
from torch.autograd import Variable

class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model)
        self.model = checkpoint['model']

        self.model.eval()

        if opt.cuda:
            self.model.cuda()
        else:
            self.model.cpu()

        self.src_dict = checkpoint['dicts']['src']['words']
        self.tgt_dict = checkpoint['dicts']['tgt']['words']

        # if opt.phrase_table.len() > 0:
        #     phraseTable = onmt.translate.PhraseTable.new(opt.phrase_table)

    def buildData(self, srcBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD) for b in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                        onmt.Constants.UNK_WORD,
                        onmt.Constants.BOS_WORD,
                        onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(
            {'words': srcData},
            {'words': tgtData} if tgtData else None,
            self.opt.batch_size, self.opt.cuda)


    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    # FIXME phrase table
                    tokens[i] = src[maxIndex[0]]

        return tokens


    def translateBatch(self, batch):
        srcBatch, tgtBatch = batch
        sourceLength = srcBatch.size(0)
        batchSize = srcBatch.size(1)
        beamSize = self.opt.beam_size

        # have to execute the encoder manually to deal with padding
        encStates = None
        context = []
        for srcBatch_t in srcBatch.chunk(srcBatch.size(0)):
            encStates, context_t = self.model.encoder(srcBatch_t, hidden=encStates)
            batchPadIdx = srcBatch_t.data.squeeze(0).eq(onmt.Constants.PAD).nonzero()
            if batchPadIdx.nelement() > 0:
                batchPadIdx = batchPadIdx.squeeze(1)
                encStates[0].data.index_fill_(1, batchPadIdx, 0)
                encStates[1].data.index_fill_(1, batchPadIdx, 0)
            context += [context_t]


        context = torch.cat(context)
        rnnSize = context.size(2)

        # Expand tensors for each beam.
        context = Variable(context.data.repeat(1, beamSize, 1))
        decStates = (Variable(encStates[0].data.repeat(1, beamSize, 1)),
                     Variable(encStates[1].data.repeat(1, beamSize, 1)))

        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]

        decOut = self.model.make_init_decoder_output(context)

        padMask = srcBatch.data.eq(onmt.Constants.PAD).t().unsqueeze(0).repeat(beamSize, 1, 1)
        def applyContextMask(m):
            if isinstance(m, onmt.modules.GlobalAttention):
                m.applyMask(padMask)

        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.opt.max_sent_length):

            self.model.decoder.apply(applyContextMask)

            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                               if not b.done]).t().contiguous().view(1, -1)

            decOut, decStates, attn = self.model.decoder(
                Variable(input), decStates, context, decOut)
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            out = self.model.generator.forward(decOut)

            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(
                        -1, beamSize, remainingSents, decState.size(2))[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # now make active contiguous
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx) \
                                    .view(*newSize))

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)
            padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch.data[:, b].ne(onmt.Constants.PAD).nonzero().squeeze(1)
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [hyps]
            allAttn += [attn]

        goldScore = 0  # FIXME
        return allHyp, allScores, allAttn, goldScore

    def translate(self, srcBatch, goldBatch):
        dataset = self.buildData(srcBatch, goldBatch)
        assert(len(dataset) == 1)  # FIXME
        batch = dataset[0]

        pred, predScore, attn, goldScore = self.translateBatch(batch)

        predBatch = []
        for b in range(batch[0].size(1)):
            predBatch.append([self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                              for n in range(self.opt.n_best)])

        return predBatch, predScore, goldScore
