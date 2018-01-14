# ZPar处理新闻分为两步，此为第二步<依赖分析>
# 对ZPar词性标注的结果进行依赖分析，提取句子中的主谓宾
save_dir="../../data/zpar/depparser/"

if [ ! -d "../../data/zpar/" ]; then
 mkdir ../../data/zpar/
fi
if [ ! -d "../../data/zpar/depparser/" ]; then
 mkdir ../../data/zpar/depparser/
fi

filelist=`find ../../data/zpar/postagger/ -type f`
for file in $filelist
do
 filename=${file##*/}
 ./../../tools/zpar/dist/english.depparser/depparser $file $save_dir$filename"_zpar_dep" ../../tools/zpar/english-models/depparser
done
