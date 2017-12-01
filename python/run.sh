echo 'Running Edge-Centric Scalable K-means'
datafolder="./data/"
inputFileName="BlogCatalog.mat"
outputFileName="SDE-$inputFileName"
dataFile="$datafolder$inputFileName"
outputFile="$datafolder$outputFileName"
repeat=1
k=5000
portion=0.9
function run(){
  python3 kmeans.py -f "$dataFile" -k "$k" -o "$outputFile"
  python3 linearSVM.py -g "$dataFile" -s "$outputFile" -p "$portion"
}
for (( i=0; i<$repeat; i++ ))
do
  run
done
