filelist=`find ../../data/processed_news/ -type f`
save_dir="../../data/zpar/pos"

if [ ! -d "../../data/zpar" ]; then
 mkdir ../../data/zpar
fi
if [ ! -d "../../data/zpar/pos" ]; then
 mkdir ../../data/zpar/pos
fi

for file in $filelist
do
 filename=${file##*/}
#  echo $save_dir$filename"_zpar_pos"
 ./../../tools/zpar/dist/english.postagger/postagger $file $save_dir$filename"_zpar_pos" ../../tools/zpar/english-models/tagger
done
