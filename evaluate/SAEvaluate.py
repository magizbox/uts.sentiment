import os
import sys
import re


def read(path):
    with open(path) as f:
        content = f.read()
    return content


class SA(object):
    def __init__(self):
        self.id = ""
        self.text = ""
        self.tag = ""
        self.aspects = list()
        self.values = list()


def load_sa(file_name):
    input_string = read(file_name).split("\n")
    sa = list()
    sentiment = SA()
    i = 0
    for i in range(len(input_string)):
        line = input_string[i]
        line = line.replace("#", "#")
        line = line.strip()
        if re.match(".?#\\d+", line):
            if sentiment is not None:
                sa.append(sentiment)
            sentiment = SA()
            sentiment.id = re.match(".?#\\d+", line).group(0)
        else:
            if re.match("\\{([A-Z]+)(.*)([a-z]+)\\}", line):
                sentiment.tag = line
                tokens = line.split(", ")
                for t in range(0, len(tokens), 2):
                    tokens[t] = tokens[t].replace("{", "")
                    tokens[t] = tokens[t].replace("}", "")
                    tokens[t] = tokens[t].strip(" ")
                    sentiment.aspects.append(tokens[t])
                for t in range(1, len(tokens), 2):
                    tokens[t] = tokens[t].replace("{", "")
                    tokens[t] = tokens[t].replace("}", "")
                    tokens[t] = tokens[t].strip(" ")
                    sentiment.values.append(tokens[t])
            elif line != "" and len(line) > 10:
                # print(line)
                sentiment.text = line
    sa.append(sentiment)
    return sa[1:]


def evaluate(gold, answer):
    all_aspects = list()
    gold_asp_count = list()
    ans_all_asp_count = list()
    ans_asp_count = list()
    ans_value_count = list()

    gold_sa = load_sa(gold)
    ans_sa = load_sa(answer)

    for i in range(len(gold_sa)):
        asp = gold_sa[i].aspects
        for j in range(len(asp)):
            if asp[j] not in all_aspects:
                all_aspects.append(asp[j])

    for i in range(len(all_aspects)):
        gold_asp_count.append(0)
        ans_asp_count.append(0)
        ans_value_count.append(0)
        ans_all_asp_count.append(0)

    for i in range(len(gold_sa)):
        asp = gold_sa[i].aspects
        for j in range(len(asp)):
            id = all_aspects.index(asp[j])
            if id != -1:
                gold_asp_count[id] = gold_asp_count[id] + 1
            else:
                print("!!! ERROR " + gold_sa[i].id)

    for i in range(len(ans_sa)):
        asp = ans_sa[i].aspects
        for j in range(len(asp)):
            id = all_aspects.index(asp[j])
            if id != -1:
                ans_all_asp_count[id] = ans_all_asp_count[id] + 1
            else:
                print("!!! Warning " + asp[j])

    for i in range(len(gold_sa)):
        g = gold_sa[i]
        a = ans_sa[i]
        if g.id != a.id:
            print("Lỗi gióng hàng:" + g.id + " <-> " + a.id)
        else:
            # if g.text != a.text:
            gasp = g.aspects
            aasp = a.aspects
            gval = g.values
            aval = a.values
            for j in range(len(gasp)):
                if gasp[j] in aasp:
                    id = aasp.index(gasp[j])
                    aspect = gasp[j]
                    gid = all_aspects.index(aspect)
                    ans_asp_count[gid] = ans_asp_count[gid] + 1
                    if not check_duplicate_asp(aspect, aasp):
                        if gval[j] == aval[id]:
                            ans_value_count[gid] = ans_value_count[gid] + 1

    print("Evaluation Result >> File:" + answer + "<> [" + gold + "]")
    # print("%30s" % " ")
    for i in range(len(all_aspects)):
        print("\t%-s" % "asp#" + str(i + 1), sep=' ', end='', flush=True)
    print()

    print("%30s" % "Gold count", sep=' ', end='', flush=True)
    for i in range(len(all_aspects)):
        print("\t%-d" % gold_asp_count[i], sep=' ', end='', flush=True)
    print()

    print("%30s" % "ANSWER count", sep=' ', end='', flush=True)
    for i in range(len(all_aspects)):
        print("\t%d" % ans_all_asp_count[i], sep=' ', end='', flush=True)
    print("\n")
    print("%30s" % "Correct ANSWER: aspect", sep=' ', end='', flush=True)
    for i in range(len(all_aspects)):
        print("\t%d" % ans_asp_count[i], sep=' ', end='', flush=True)
    print()
    print("%30s" % "Precision: aspect", sep=' ', end='', flush=True)
    for i in range(len(all_aspects)):
        if ans_all_asp_count[i] > 0:
            p = ans_asp_count[i] / ans_all_asp_count[i]
        print("\t{:.2f}".format(p), sep=' ', end='', flush=True)
    print()
    print("%30s" % "Recall: aspect", sep=' ', end='', flush=True)
    for i in range(len(all_aspects)):
        if gold_asp_count[i] > 0:
            r = ans_asp_count[i] / gold_asp_count[i]
        print("\t{:.2f}".format(r), sep=' ', end='', flush=True)
    print()
    print("%30s" % "F1 score: aspect", sep=' ', end='', flush=True)
    for i in range(len(all_aspects)):
        if ans_all_asp_count[i] > 0:
            p = ans_asp_count[i] / ans_all_asp_count[i]
        if ans_all_asp_count[i] > 0:
            r = ans_asp_count[i] / gold_asp_count[i]
        if p + r > 0:
            f = 2.0 * (p * r) / (p + r)
        print("\t{:.2f}".format(f), sep=' ', end='', flush=True)

    tgold = 0
    for i in range(len(gold_asp_count)):
        tgold = tgold + gold_asp_count[i]

    tans = 0
    for i in range(len(ans_all_asp_count)):
        tans = tans + ans_all_asp_count[i]

    tcans = 0
    for i in range(len(ans_asp_count)):
        tcans = tcans + ans_asp_count[i]

    tvalue = 0
    for i in range(len(ans_value_count)):
        tvalue = tvalue + ans_value_count[i]

    p = tcans / tans
    r = tcans / tgold
    f1 = 2 * p * r / (p + r)
    print()
    print("%30s" % "Over All ANSWER: aspect:----", sep=' ', end='', flush=True)
    print("\tPrecision = {:.2f}\tRecall = {:.2f}\tF1 score = {:.2f}".format(p, r, f1))
    print()
    print("%30s" % "Correct ANSWER: aspect,value", sep=' ', end='', flush=True)
    for i in range(len(all_aspects)):
        print("\t%d" % ans_value_count[i], sep=' ', end='', flush=True)
    print()

    print("%30s" % "Precision: aspect, value", sep=' ', end='', flush=True)
    for i in range(len(all_aspects)):
        if ans_all_asp_count[i] > 0:
            p = ans_value_count[i] / ans_all_asp_count[i]
        print("\t{:.2f}".format(p), sep=' ', end='', flush=True)
    print()

    print("%30s" % "Recall: aspect, value", sep=' ', end='', flush=True)
    for i in range(len(all_aspects)):
        if ans_all_asp_count[i] > 0:
            r = ans_value_count[i] / gold_asp_count[i]
        print("\t{:.2f}".format(r), sep=' ', end='', flush=True)
    print()

    print("%30s" % "F1 score: aspect, value", sep=' ', end='', flush=True)
    for i in range(len(all_aspects)):
        if ans_all_asp_count[i] > 0:
            p = 1.0 * ans_value_count[i] / ans_all_asp_count[i]
        if gold_asp_count[i] > 0:
            r = 1.0 * ans_value_count[i] / gold_asp_count[i]
        if p + r > 0:
            f = 2.0 * (p * r) / (p + r)
        print("\t{:.2f}".format(f), sep=' ', end='', flush=True)
    print()

    p = 1.0 * tvalue / tans
    r = 1.0 * tvalue / tgold
    f1 = 2 * p * r / (p + r)
    print()
    print("%30s" % "Over All ANSWER: aspect, value:----", sep=' ', end='', flush=True)
    print("\tPrecision = {:.2f}\tRecall = {:.2f}\tF1 score = {:.2f}".format(p, r, f1))
    print("\n")

    for i in range(len(all_aspects)):
        print("asp#" + str(i + 1) + ": " + all_aspects[i])


def check_duplicate_asp(asp, asp_list):
    count = 0
    for i in range(len(asp_list)):
        if asp_list[i] == asp:
            count += 1
    if count > 1:
        return True
    return False


def evaluate_folder(gold, answer):
    gold_files = os.listdir(gold)
    ans_files = os.listdir(answer)

    for i in range(len(ans_files)):
        af = ans_files[i].lower()
        if 'hotel' in af:
            evaluate(gold + "/" + gold_files[0], answer + "/" + ans_files[i])
        elif "restaurant" in af:
            evaluate(gold + "/" + gold_files[1], answer + "/" + ans_files[i])


if __name__ == '__main__':
    orig_stdout = sys.stdout
    f = open('report.xlsx', 'w')
    sys.stdout = f
    evaluate_folder('SA_Test', "SA_Evaluate")
    sys.stdout = orig_stdout
    f.close()
