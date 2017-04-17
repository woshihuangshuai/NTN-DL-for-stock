filelist=`find ../../data/zpar/pos -type f`
save_dir="../../data/zpar/dep"

if [ ! -d "../../data/zpar" ]; then
 mkdir ../../data/zpar
fi
if [ ! -d "../../data/zpar/dep" ]; then
 mkdir ../../data/zpar/dep
fi

for file in $filelist
do
 filename=${file##*/}
#  echo $save_dir$filename"_zpar_dep"
 ./../../tools/zpar/dist/english.depparser/depparser $file $save_dir$filename"_zpar_dep" ../../tools/zpar/english-models/depparser
done
