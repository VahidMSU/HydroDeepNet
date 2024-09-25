@echo off

start cmd /c "aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2014.h5 E:/NSRDB/nsrdb_2014_full.h5 --no-sign-request --region us-east-2"