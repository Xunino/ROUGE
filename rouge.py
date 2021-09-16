import collections


class ROUGE:
    """
        ROUGE: A Package for Automatic Evaluation of Summaries
        https://aclanthology.org/W04-1013.pdf
    """

    def __init__(self):
        pass

    def _create_ngrams(self, token, n=3):
        token = token.split()
        len_token = len(token)
        ngrams = collections.defaultdict(int)
        for i in range(len_token - n + 1):
            ngrams[" ".join(token[i:i + n])] += 1
        return ngrams

    def _recall_safe(self, x, y):
        return max(x / y, 0)

    def _precision_safe(self, x, y):
        return max(x / y, 0)

    def _f_score_save(self, recall, precision, beta=1):
        if (recall + precision) > 0:
            f_score = max((1 + beta ** 2) * recall * precision / (recall + (beta ** 2) * precision), 0)
        else:
            f_score = 0.0
        return f_score

    def _lcs_len(self, x, y):
        """
            This function returns length of longest common sequence of x and y.
        """

        if len(x) == 0 or len(y) == 0:
            return 0

        xx = x[:-1]  # xx = sequence x without its last element
        yy = y[:-1]

        if x[-1] == y[-1]:  # if last elements of x and y are equal
            return self._lcs_len(xx, yy) + 1
        else:
            return max(self._lcs_len(xx, y), self._lcs_len(x, yy))

    def calculate_ngrams(self, candidates, references, n=3):
        for candidate, reference in zip(candidates, references):
            # Do check matches
            target_ngrams = self._create_ngrams(reference, n)
            pred_ngrams = self._create_ngrams(candidate, n)

            matches = 0
            for ngram in target_ngrams.keys():
                matches += min(target_ngrams[ngram], pred_ngrams[ngram])

            # Do recall
            recall = self._recall_safe(matches, len(target_ngrams))

            # Do precision
            precision = self._precision_safe(matches, len(pred_ngrams))

            # Do f_score
            f_score = self._f_score_save(recall, precision)
            return recall, precision, f_score

    def calculate_lcs(self, candidates, references, beta=8):
        for candidate, reference in zip(candidates, references):
            candidate = candidate.split()
            reference = reference.split()

            # Do longest common sentence length
            lcs_length = self._lcs_len(candidate, reference)

            # Do recall
            recall = self._recall_safe(lcs_length, len(reference))

            # Do precision
            precision = self._precision_safe(lcs_length, len(candidate))

            # Do f_score
            f_score = f_score = self._f_score_save(recall, precision, beta)
            return recall, precision, f_score


if __name__ == '__main__':
    candidates = ["Hello the world He can see nothing"]
    references = ["Hello the world I can show everything"]

    rouge_l = ROUGE().calculate_lcs(candidates, references, 8)
    print(rouge_l)

    rouge_3 = ROUGE().calculate_ngrams(candidates, references, n=3)
    print(rouge_3)
