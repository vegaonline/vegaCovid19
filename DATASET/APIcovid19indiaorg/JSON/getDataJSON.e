wget -c https://api.covid19india.org/raw_data1.json
wget -c https://api.covid19india.org/raw_data2.json
wget -c https://api.covid19india.org/raw_data3.json
wget -c https://api.covid19india.org/data.json
wget -c https://api.covid19india.org/state_district_wise.json
wget -c https://api.covid19india.org/states_daily.json
wget -c https://api.covid19india.org/state_test_data.json
wget -c https://api.covid19india.org/districts_daily.json
wget -c https://api.covid19india.org/zones.json
wget -c https://api.covid19india.org/resources/resources.json
dos2unix *
mv data.json national_time_series_stats_test_counts.json
