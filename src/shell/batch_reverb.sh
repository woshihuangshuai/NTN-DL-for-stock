filelist=`find ../../data/processed_news/ -type f`
save_dir="../../data/reverb/"

if [ ! -d "../../data/reverb/" ]; then
 mkdir ../../data/reverb/
fi

for file in $filelist
do
 filename=${file##*/}
#  echo $save_dir$filename"_reverb"
 java -Xmx512m -jar ../../tools/reverb/reverb-latest.jar $file > $save_dir$filename"_reverb"
done
