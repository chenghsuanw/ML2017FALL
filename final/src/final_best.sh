#!/bin/bash
python3 emb.py $1
python3 test.py $1 $2 
