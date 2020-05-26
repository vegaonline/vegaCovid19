#!/bin/bash
echo " Present directory: `pwd`"
(
cd APIcovid19indiaorg
cd CSV
echo " Present directory: `pwd`"
echo " ............. Downloading Indian CSV data.."
./getDataCSV.e
cd ../JSON
echo " Present directory: `pwd`"
echo " ............. Downloading Indian JSON data.."
./getDataJSON.e
)
(
cd JohnHopkinsU_CSSE
echo " Present directory: `pwd`"
echo " ............. Downloading Global data.."
git pull
)
echo " Present directory: `pwd`"
./fixIndFile.e
