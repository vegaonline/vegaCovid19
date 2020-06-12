mkdir OLDCSV
mv *.csv OLDCSV/
wget -c https://api.covid19india.org/csv/latest/raw_data1.csv
wget -c https://api.covid19india.org/csv/latest/raw_data2.csv
wget -c https://api.covid19india.org/csv/latest/raw_data3.csv
wget -c https://api.covid19india.org/csv/latest/raw_data4.csv
wget -c https://api.covid19india.org/csv/latest/raw_data5.csv
wget -c https://api.covid19india.org/csv/latest/death_and_recovered1.csv
wget -c https://api.covid19india.org/csv/latest/death_and_recovered2.csv
wget -c https://api.covid19india.org/csv/latest/state_wise.csv 
wget -c https://api.covid19india.org/csv/latest/case_time_series.csv
wget -c https://api.covid19india.org/csv/latest/district_wise.csv
wget -c https://api.covid19india.org/csv/latest/state_wise_daily.csv 
wget -c https://api.covid19india.org/csv/latest/statewise_tested_numbers_data.csv
wget -c https://api.covid19india.org/csv/latest/tested_numbers_icmr_data.csv
wget -c https://api.covid19india.org/csv/latest/sources_list.csv
dos2unix *
