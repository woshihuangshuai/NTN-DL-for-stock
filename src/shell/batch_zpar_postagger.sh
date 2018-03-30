# ZPar处理新闻分为两步，此为第一步<词性标注>
# 对合并后的新闻文件进行ZPar的词性标注操作

save_dir="../../data/zpar/postagger/"

if [ ! -d "../../data/zpar/" ]; then
 mkdir ../../data/zpar/
fi
if [ ! -d "../../data/zpar/postagger/" ]; then
 mkdir ../../data/zpar/postagger/
fi

filelist=`find ../../data/merged_processed_news/ -type f`
for file in $filelist
do
 filename=${file##*/}
 ./../../tools/zpar/dist/english.postagger/tagger $file $save_dir$filename"_zpar_pos" ../../tools/zpar/english-models/tagger
done
