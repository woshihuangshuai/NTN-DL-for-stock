# 将一个日期目录下的所有新闻，合并为同一个文件，以加速ReVerb和ZPar的速度
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
