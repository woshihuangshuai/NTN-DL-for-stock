# 批量对合并后的原始新闻文件进行ReVerb处理，以句子为单位提取句子中的三元候选元组，实验证明使用原始新闻提取的结果更好
save_dir="../../data/reverb/"

if [ ! -d "../../data/reverb/" ]; then
 mkdir ../../data/reverb/
fi

filelist=`find ../../data/merged_raw_news/ -type f`
for file in $filelist
do
 filename=${file##*/}
 java -Xmx2048m -jar ../../tools/reverb/reverb-latest.jar $file > $save_dir$filename"_reverb"
done
