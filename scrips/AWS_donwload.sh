#!/bin/bash

aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2006.h5 /data/NSRDB/nsrdb_2006_full.h5 --no-sign-request --region us-east-2
aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2007.h5 /data/NSRDB/nsrdb_2007_full.h5 --no-sign-request --region us-east-2
aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2008.h5 /data/NSRDB/nsrdb_2008_full.h5 --no-sign-request --region us-east-2
aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2009.h5 /data/NSRDB/nsrdb_2009_full.h5 --no-sign-request --region us-east-2
aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2010.h5 /data/NSRDB/nsrdb_2010_full.h5 --no-sign-request --region us-east-2
aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2011.h5 /data/NSRDB/nsrdb_2011_full.h5 --no-sign-request --region us-east-2
aws s3 cp s3://nrel-pds-nsrdb/v3/nsrdb_2012.h5 /data/NSRDB/nsrdb_2012_full.h5 --no-sign-request --region us-east-2


