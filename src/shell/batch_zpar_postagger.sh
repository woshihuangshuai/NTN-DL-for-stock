save_dir="../../data/zpar/postagger/"

if [ ! -d "../../data/zpar/" ]; then
 mkdir ../../data/zpar/
fi
if [ ! -d "../../data/zpar/postagger/" ]; then
 mkdir ../../data/zpar/postagger/
fi

filelist=`find ../../data/merged_news/ -type f`
for file in $filelist
do
 filename=${file##*/}
 ./../../tools/zpar/dist/english.postagger/tagger $file $save_dir$filename"_zpar_pos" ../../tools/zpar/english-models/tagger
done
