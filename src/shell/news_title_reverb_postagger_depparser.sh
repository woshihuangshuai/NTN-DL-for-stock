# ReVerb
source_file="../../data/news_title/reuters_news_title.txt"
result_file="../../data/news_title/reuters_news_title_reverb"
`java -Xmx2048m -jar ../../tools/reverb/reverb-latest.jar $source_file > $result_file`

source_file="../../data/news_title/bloomberg_news_title.txt"
result_file="../../data/news_title/bloomberg_news_title_reverb"
`java -Xmx2048m -jar ../../tools/reverb/reverb-latest.jar $source_file > $result_file`

# ZPar postagger
source_file="../../data/news_title/reuters_news_title.txt"
result_file="../../data/news_title/reuters_news_title_zpar_postagger"
`./../../tools/zpar/dist/english.postagger/tagger $source_file $result_file ../../tools/zpar/english-models/tagger`

source_file="../../data/news_title/bloomberg_news_title.txt"
result_file="../../data/news_title/bloomberg_news_title_zpar_postagger"
`./../../tools/zpar/dist/english.postagger/tagger $source_file $result_file ../../tools/zpar/english-models/tagger`

# ZPar depparser
source_file="../../data/news_title/reuters_news_title_zpar_postagger"
result_file="../../data/news_title/reuters_news_title_zpar_depparser"
`./../../tools/zpar/dist/english.depparser/depparser $file $save_dir$filename"_zpar_dep" ../../tools/zpar/english-models/depparser`

source_file="../../data/news_title/bloomberg_news_title_zpar_postagger"
result_file="../../data/news_title/bloomberg_news_title_zpar_depparser"
`./../../tools/zpar/dist/english.depparser/depparser $file $save_dir$filename"_zpar_dep" ../../tools/zpar/english-models/depparser`
