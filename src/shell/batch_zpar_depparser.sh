news_folders=(bloomberg reuters)
for folder in ${news_folders[@]}
do
 dirlist=`find ../../data/zpar/postagger/$folder/* -type d`

 if [ ! -d "../../data/zpar/" ]; then
  mkdir ../../data/zpar/
 fi
 if [ ! -d "../../data/zpar/depparser/" ]; then
  mkdir ../../data/zpar/depparser/
 fi
 if [ ! -d "../../data/zpar/depparser/$folder/" ]; then
  mkdir ../../data/zpar/depparser/$folder/
 fi

 for dir in $dirlist
 do
  dir_idx=${dir##*/}
  save_dir="../../data/zpar/depparser/$folder/$dir_idx/"
  echo $save_dir
  if [ ! -d "../../data/zpar/depparser/$folder/$dir_idx/" ]; then
   mkdir ../../data/zpar/depparser/$folder/$dir_idx/
  fi

  filelist=`find $dir/ -type f`

  for file in $filelist
  do
   filename=${file##*/}
#    echo $file
#    echo $save_dir$filename"_zpar_pos"
   ./../../tools/zpar/dist/english.depparser/depparser $file $save_dir$filename"_zpar_dep" ../../tools/zpar/english-models/depparser
  done
 done
done
