TEXT= $raw_data_dir
python preprocess.py --source-lang art --target-lang abs \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir  $data_dir\
    --workers 64 --joined-dictionary
