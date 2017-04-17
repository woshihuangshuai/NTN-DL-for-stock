filelist=`find ../../data/processed_news/ -type f`
save_dir="../../data/processed_news/reverb/"

if [ ! -d "../../data/processed_news/reverb/" ]; then
 mkdir ../../data/processed_news/reverb/
fi

for file in $filelist
do
 filename=${file##*/}
#  echo $save_dir$filename"_reverb"
 java -Xmx512m -jar ../../tools/reverb/reverb-latest.jar $file|$save_dir$filename"_reverb"
done
