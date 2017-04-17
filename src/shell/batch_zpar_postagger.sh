filelist=`find ../../data/processed_news/ -type f`
save_dir="../../data/processed_news/zpar/pos"

if [ ! -d "../../data/processed_news/zpar" ]; then
 mkdir ../../data/processed_news/zpar
fi
if [ ! -d "../../data/processed_news/zpar/pos" ]; then
 mkdir ../../data/processed_news/zpar/pos
fi

for file in $filelist
do
 filename=${file##*/}
 echo $save_dir$filename"_zpar_pos"
done
