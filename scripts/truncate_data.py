import os
import re


def process(filename, f_type, linelength=None):
    lines = []
    num = 0
    with open(filename, 'r') as reader:
        for line in reader:
            line = line.strip()
            if f_type == 'abs':
                sen_list = re.split("<t>|</t>", line)
                sen_list = [sen.strip() for sen in sen_list if len(sen.strip()) > 0]
                line = ' '.join(sen_list)
            if linelength is not None and len(line.split()) > linelength:
                num += 1
                line = ' '.join(line.split()[:linelength])
            lines.append(line)
    print('{} of line larger than linelength'.format(num))
    with open(os.path.join('truncate_500', filename), 'w') as writer:
        writer.write('\n'.join(lines))


if __name__ == "__main__":
    os.makedirs('truncate_500', exist_ok=True)
    dataset = ['train', 'valid', 'test']
    src = 'art'
    tgt = 'abs'
    for onedata in dataset:
        process(onedata + "." + src, src, 500)
        process(onedata + "." + tgt, tgt)
        print('{} done'.format(onedata))
