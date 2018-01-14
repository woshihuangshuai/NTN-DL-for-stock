# 批量对合并后的新闻文件进行ReVerb处理，以句子为单位提取句子中的三元候选元组
save_dir="../../data/reverb/"

if [ ! -d "../../data/reverb/" ]; then
 mkdir ../../data/reverb/
fi

filelist=`find ../../data/merged_news/ -type f`
for file in $filelist
do
 filename=${file##*/}
 java -Xmx2048m -jar ../../tools/reverb/reverb-latest.jar $file > $save_dir$filename"_reverb"
done
