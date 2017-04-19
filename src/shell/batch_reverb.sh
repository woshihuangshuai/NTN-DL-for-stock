news_folders=(bloomberg reuters)
for folder in ${news_folders[@]}
do
 dirlist=`find ../../data/processed_news/$folder/* -type d`
 save_dir="../../data/reverb/$folder/"

 if [ ! -d "../../data/reverb/" ]; then
  mkdir ../../data/reverb/
 fi
 if [ ! -d "../../data/reverb/$folder/" ]; then
  mkdir ../../data/reverb/$folder/
 fi

 for dir in $dirlist
 do
  dir_idx=${dir##*/}
  echo $save_dir$dir_idx"_reverb"
  find $dir"/" -type f | java -Xmx1024m -jar ../../tools/reverb/reverb-latest.jar -f > $save_dir$dir_idx"_reverb"
 done
done
