# filelist=`find ~/Documents/NTN-DL-for-stock/data/processed_news/ -type f`
filelist=`find ~/Documents/NTN-DL-for-stock/data/bloomberg_and_reuters_news_title/ -type f`
for file in $filelist
do
 # java -Xmx512m -jar ~/Documents/NTN-DL-for-stock/tools/reverb/reverb-latest.jar $file
 filename=${file##*/}
 echo $filename"aaaaa"
done
