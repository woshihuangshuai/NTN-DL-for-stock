filelist=`find ../../data/processed_news/zpar/pos -type f`
save_dir="../../data/processed_news/zpar/dep"

if [ ! -d "../../data/processed_news/zpar" ]; then
 mkdir ../../data/processed_news/zpar
fi
if [ ! -d "../../data/processed_news/zpar/dep" ]; then
 mkdir ../../data/processed_news/zpar/dep
fi

for file in $filelist
do
 filename=${file##*/}
 echo $save_dir$filename"_zpar_dep"
done
