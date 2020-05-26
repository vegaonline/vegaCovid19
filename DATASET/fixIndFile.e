#!/bin/bash
while IFS=, read -r col1 col2 col3 col4 col5 col6 col7 ;
do
	if [ "$col1" = "Date" ] ; then
		colVal=$col1
	else
		colVal=$col1"2020"
	fi
	echo "$colVal,$col2,$col3,$col4,$col5,$col6,$col7"
done < APIcovid19indiaorg/CSV/case_time_series.csv  > APIcovid19indiaorg/CSV/case_time_series_1.csv
while IFS=, read -r col1 col2 col3 col4 col5 col6 col7 ;
do
	if [ "$col1" = "Date" ] ; then
		colVal=$col1
	else
		colVal=$col1"2020"
	fi
	echo "$colVal,$col2,$col3,$col4,$col5,$col6,$col7"
done <<< $(tail -1 APIcovid19indiaorg/CSV/case_time_series.csv) >> APIcovid19indiaorg/CSV/case_time_series_1.csv

