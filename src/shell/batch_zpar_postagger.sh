news_folders=(bloomberg reuters)
for folder in ${news_folders[@]}
do
 dirlist=`find ../../data/processed_news/$folder/* -type d`
 
 if [ ! -d "../../data/zpar/" ]; then
  mkdir ../../data/zpar/
 fi
 if [ ! -d "../../data/zpar/postagger/" ]; then
  mkdir ../../data/zpar/postagger/
 fi
 if [ ! -d "../../data/zpar/postagger/$folder/" ]; then
  mkdir ../../data/zpar/postagger/$folder/
 fi

 for dir in $dirlist
 do
  dir_idx=${dir##*/}
  save_dir="../../data/zpar/postagger/$folder/$dir_idx/"
  echo $save_dir
  if [ ! -d "../../data/zpar/postagger/$folder/$dir_idx/" ]; then
   mkdir ../../data/zpar/postagger/$folder/$dir_idx/
  fi

  filelist=`find ../../data/processed_news/$folder/$dir_idx/ -type f`

  for file in $filelist
  do
   filename=${file##*/}
#    echo $file
#    echo $save_dir$filename"_zpar_pos"
   ./../../tools/zpar/dist/english.postagger/tagger $file $save_dir$filename"_zpar_pos" ../../tools/zpar/english-models/tagger
  done
 done
done
