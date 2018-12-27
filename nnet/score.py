#%%
import argparse
from collections import defaultdict
#%%

def labelfile(filen):
    utttolabel = {}
    with open(filen, 'r') as wp:
        for line in wp:
            utt, label = line.split()[:2]
            utttolabel[utt] = label
    return utttolabel


def labeltoscore(labels, scores):
    labscore = defaultdict(list)
    logwarns = 0
    for utt, label in labels.items():
        if not utt in scores:
            logwarns = logwarns + 1
            print("Utterance %s not found in scores" % (utt))
            continue
        score = scores[utt]
        labscore[label].append(score)
    if logwarns > 0:
        print("Encountered %i errors" % (logwarns))
    return labscore

def scorefile(filen):
    utttoscore = {}
    with open(filen, 'r') as wp:
        for line in wp:
            utt, score = line.split()[:2]
            utt = utt.split("/")[-1]
            utttoscore[utt] = float(score)
    return utttoscore

#%%
def compute_eer(scores):
    target_scores = sorted(scores["genuine"])
    nontarget_scores = sorted(scores["spoof"])

    tgt_size = len(target_scores)
    ntgt_size = len(nontarget_scores)
    tgt_pos = 0
    ntgt_pos = ntgt_size-1
    while nontarget_scores[ntgt_pos] >= target_scores[tgt_pos]:
        tgt_pos += 1
        ntgt_pos = max(0, ntgt_size -1 - int(float(tgt_pos)/tgt_size*ntgt_size))
    
    eer = float(tgt_pos)/tgt_size 
    thresh = target_scores[tgt_pos]
    # print("EER : {0:.4%} at threshold {1:.4f}, tgt_pos: {2}, ntgt_pos:{3} ".format(float(tgt_pos)/tgt_size, target_scores[tgt_pos], tgt_pos, ntgt_pos))
    return eer, thresh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-score_file", type=str, default=r"result\cnn\cqcc\eval_score.txt")
    parser.add_argument("-label_file", type=str, default="eval_label.txt")
    args = parser.parse_args()
    scores= labeltoscore(labelfile(args.label_file),scorefile(args.score_file))
    eer, thresh = compute_eer(scores)
    print("Evaluation set :EER = {0:.2%} at threshold = {1:.4f} ".format(eer, thresh))


if __name__ == '__main__':
    main()
