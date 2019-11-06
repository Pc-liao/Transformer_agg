TEXT=/chome/lpc/cnndm_data/truncate_500_vocab_3w/segmented/
python preprocess.py --source-lang art --target-lang abs \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir /chome/lpc/cnndm_data/truncate_500_vocab_3w/cnndm_bin_joindir \
    --workers 64 --joined-dictionary
