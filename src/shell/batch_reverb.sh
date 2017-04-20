save_dir="../../data/reverb/"

if [ ! -d "../../data/reverb/" ]; then
 mkdir ../../data/reverb/
fi

filelist = `find ../../data/merged_news/ -type f`
for file in $filelist
do
 filename=${file##*/}
 java -Xmx1024m -jar ../../tools/reverb/reverb-latest.jar > $save_dir$filename"_reverb"
done
