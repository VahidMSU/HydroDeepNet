#!/bin/bash

aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2020.h5 /data/NSRDB/nsrdb_2020_full.h5 --no-sign-request --region us-east-2
aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2019.h5 /data/NSRDB/nsrdb_2019_full.h5 --no-sign-request --region us-east-2
aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2018.h5 /data/NSRDB/nsrdb_2018_full.h5 --no-sign-request --region us-east-2
aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2015.h5 /data/NSRDB/nsrdb_2015_full.h5 --no-sign-request --region us-east-2