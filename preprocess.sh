TEXT= (your dataset dir)
python preprocess.py --source-lang art --target-lang abs \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir  (dataset outdir)\
    --workers 64 --joined-dictionary
