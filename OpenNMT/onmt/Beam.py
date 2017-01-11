# Class for managing the internals of the beam search process.
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

import torch
import onmt


class Beam(object):
    def __init__(self, size, cuda=False):

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = [self.tt.LongTensor(size).fill_(1)]

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(onmt.Constants.PAD)]
        self.nextYs[0][0] = onmt.Constants.BOS

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def getCurrentState(self):
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def getCurrentOrigin(self):
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`. Compute and update the beam search.
    #
    # Parameters.
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns. True if beam search is complete.
    def advance(self, wordLk, attnOut):
        numWords = wordLk.size(1)

        # Sum the previous scores.
        for k in range(self.size):
            wordLk[k].add(self.scores[k])

        flatWordLk = wordLk.view(-1)

        bestScores, bestScoresId = flatWordLk.topk(self.size, 0, True, True)

        self.scores = bestScores
        self.prevKs.append(self.tt.LongTensor(
            [scoreId // numWords for scoreId in bestScoresId]))
        self.nextYs.append(self.tt.LongTensor(
            [scoreId % numWords for scoreId in bestScoresId]))
        self.attn.append(
            [attnOut[scoreId // numWords] for scoreId in bestScoresId])

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == onmt.Constants.EOS:
            self.done = True

        return self.done

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def getBest(self):
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def getHyp(self, k):
        hyp, attn = [], []

        for j in range(len(self.prevKs) - 1, 0, -1):
            hyp.append(self.nextYs[j][k])
            attn.append(self.attn[j - 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1], torch.stack(attn[::-1])
