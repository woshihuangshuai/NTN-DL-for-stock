filelist=`find ~/Documents/NTN-DL-for-stock/src/ -type f`
for file in $filelist
do
 java -Xmx512m -jar ~/Documents/NTN-DL-for-stock/tools/reverb/reverb-latest.jar $file
done
