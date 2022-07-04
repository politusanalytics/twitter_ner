# written to logs/log.txt
# echo "****** dbmdz/bert-base-turkish-cased ******"
# echo "*** wiki-ann ***"
# python train_flair.py /home/omutlu/twitter_ner/data/wiki-ann/ dbmdz/bert-base-turkish-cased
# echo "*** milliyet ***"
# python train_flair.py /home/omutlu/twitter_ner/data/milliyet-ner/ dbmdz/bert-base-turkish-cased
# echo "*** combined ***"
# python train_flair.py /home/omutlu/twitter_ner/data/combined/ dbmdz/bert-base-turkish-cased

# echo "****** cardiffnlp/twitter-xlm-roberta-base ******"
# echo "*** wiki-ann ***"
# python train_flair.py /home/omutlu/twitter_ner/data/wiki-ann/ cardiffnlp/twitter-xlm-roberta-base
# echo "*** milliyet ***"
# python train_flair.py /home/omutlu/twitter_ner/data/milliyet-ner/ cardiffnlp/twitter-xlm-roberta-base
# echo "*** combined ***"
# python train_flair.py /home/omutlu/twitter_ner/data/combined/ cardiffnlp/twitter-xlm-roberta-base



# written to logs/twitter_log.txt
# echo "****** cardiffnlp/twitter-xlm-roberta-base ******"
# echo "*** wiki-ann ***"
# python train.py cardiffnlp/twitter-xlm-roberta-base wiki-ann 42
# echo "*** milliyet ***"
# python train.py cardiffnlp/twitter-xlm-roberta-base milliyet-ner 42
# echo "*** combined ***"
# python train.py cardiffnlp/twitter-xlm-roberta-base combined 42
