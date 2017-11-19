echo 'Running Edge-Centric Scalable K-means'
datafolder="./data/"
inputFileName="BlogCatalog.mat"
outputFileName="SDE-$inputFileName"
dataFile="$datafolder$inputFileName"
outputFile="$datafolder$outputFileName"
k=500
portion=0.9
function run(){
  python3 kmeans.py -f "$dataFile" -k "$k" -o "$outputFile"
  python3 linearSVM.py -g "$dataFile" -s "$outputFile" -p "$portion"
}
run
#for i in {1..100}; do
#  run
#done
