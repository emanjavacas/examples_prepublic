

class Translator(object):
    def __init__(self, opt):
        opt = args

        checkpoint = torch.load(opt.model)
        self.model = checkpoint['model']

        self.model.evaluate()

        if opt.cuda:
            self.model.cuda()
        else:
            self.model.cpu()

        self.src_dict = checkpoint['dicts']['src']['words']
        self.tgt_dict = checkpoint['dicts']['tgt']['words']

        # if opt.phrase_table.len() > 0:
        #     phraseTable = onmt.translate.PhraseTable.new(opt.phrase_table)

    def buildData(self, srcBatch, goldBatch):
        srcData =  [self.src_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD) for b in srcBatch]
        goldData = [self.tgt_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD,
                    onmt.Constants.BOS_WORD,
                    onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.data.Dataset.new(srcData, tgtData)


    def buildTargetTokens(pred, src, attn):
        tokens = tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS

        if self.opt.replace_unk:
            for i in range len(tokens):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(1)
                    # FIXME phrase table
                    tokens[i] = src[maxIndex[1]]

        return tokens


    def translateBatch(batch):
        sourceLength = batch[0].size(0)
        batchSize = batch[0].size(1)

        # FIXME FIXME FIXME
        self.model.maskPadding()

        encStates, context = model.encoder(batch)
        rnnSize = context.size(2)
        # if batch.targetInput != None:
        #     if batchSize > 1:
        #         models.decoder.maskPadding(batch.sourceSize, batch.sourceLength)
        #
        #     goldScore = models.decoder.computeScore(batch, encStates, context)

        # Expand tensors for each beam.
        context = context.repeatTensor(self.opt.beam_size, 1, 1)
        encStates = encStates.repeatTensor(1, self.opt.beam_size, 1)

        beam = [onmt.translate.Beam(opt.beam_size) for k in range(batchSize)]
        i = 1

        decStates = encStates

        while remainingSents > 0 and i < self.opt.max_sent_length:
            i = i + 1

            # Prepare decoder input.
            input = torch.IntTensor(opt.beam_size, remainingSents)
            sourceSizes = torch.IntTensor(remainingSents)

            for b in range(batchSize):
                if not beam[b].done:
                    sourceSizes[b] = batch.sourceSize[b]

                    # Get current state of the beam search.
                    input[:, b].copy(beam[b].getCurrentState())

            input = input.view(opt.beam_size * remainingSents)

            # if batchSize > 1:
            #     models.decoder.maskPadding(sourceSizes, batch.sourceLength, opt.beam_size)

            decOut = model.make_init_output(input)
            decOut, decStates = self.model.decoder(input, decStates, context, decOut)

            out = models.decoder.generator.forward(decOut)

            for j = 1, len(out):
                out[j] = out[j].view(opt.beam_size, remainingSents, out[j]:size(2)):transpose(1, 2):contiguous()

            wordLk = out[1]

            softmaxOut = models.decoder.softmaxAttn.output.view(opt.beam_size, remainingSents, -1)
            newRemainingSents = remainingSents

            for b = 1, batchSize:
                if not beam[b].done:
                    idx = batchIdx[b]


                    if beam[b].advance(wordLk[idx], softmaxOut[{{}, idx}]):
                        newRemainingSents = newRemainingSents - 1
                        batchIdx[b] = 0

                    for j = 1, len(decStates):
                        view = decStates[j]
                            .view(opt.beam_size, remainingSents, checkpoint.options.rnn_size)
                        view[{{}, idx}] = view[{{}, idx}].index(1, beam[b]:getCurrentOrigin())

            if newRemainingSents > 0 and newRemainingSents != remainingSents:
                # Update sentence indices within the batch and mark sentences to keep.
                toKeep = {}
                newIdx = 1
                for b = 1, len(batchIdx):
                    idx = batchIdx[b]
                    if idx > 0:
                        table.insert(toKeep, idx)
                        batchIdx[b] = newIdx
                        newIdx = newIdx + 1

                toKeep = torch.LongTensor(toKeep)

                # Update rnn states and context.
                for j = 1, len(decStates):
                    decStates[j] = decStates[j]
                        .view(opt.beam_size, remainingSents, checkpoint.options.rnn_size)
                        .index(2, toKeep)
                        .view(opt.beam_size*newRemainingSents, checkpoint.options.rnn_size)


                decOut = decOut
                    .view(opt.beam_size, remainingSents, checkpoint.options.rnn_size)
                    .index(2, toKeep)
                    .view(opt.beam_size*newRemainingSents, checkpoint.options.rnn_size)

                context = context
                    .view(opt.beam_size, remainingSents, batch.sourceLength, checkpoint.options.rnn_size)
                    .index(2, toKeep)
                    .view(opt.beam_size*newRemainingSents, batch.sourceLength, checkpoint.options.rnn_size)

            remainingSents = newRemainingSents

        for b = 1, batchSize:
            scores, ks = beam[b].sortBest()

            for n = 1, opt.n_best:
                hyp, feats, attn = beam[b].getHyp(ks[n])

                # remove unnecessary values from the attention vectors
                for j = 1, len(attn):
                    size = batch.sourceSize[b]
                    attn[j] = attn[j].narrow(1, batch.sourceLength - size + 1, size)


                table.insert(hypBatch, hyp)
                if len(feats) > 0:
                    table.insert(featsBatch, feats)

                table.insert(attnBatch, attn)
                table.insert(scoresBatch, scores[n])


            table.insert(allHyp, hypBatch)
            table.insert(allFeats, featsBatch)
            table.insert(allAttn, attnBatch)
            table.insert(allScores, scoresBatch)


        return allHyp, allFeats, allScores, allAttn, goldScore


    def translate(srcBatch, goldBatch):
        data = buildData(srcBatch, goldBatch)
        batch = data.getBatch()

        pred, predScore, attn, goldScore = translateBatch(batch)

        predBatch = []
        for b in range(batchSize):
            predBatch.append([buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                              for n in range(self.opt.n_best)])

        return predBatch, predScore, goldScore
