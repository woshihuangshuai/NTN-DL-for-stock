dirlist=`find ../../data/processed_news/* -type d`
save_dir="../../data/merged_news/"

if [ ! -d "../../data/merged_news/" ]; then
 mkdir ../../data/merged_news/
fi

for dir in $dirlist
do
 filename=${dir##*/}
 echo $dir
 filelist=`find $dir/ -type f`
 for file in $filelist
 do
  cat $file >> $save_dir$filename
 done
done
