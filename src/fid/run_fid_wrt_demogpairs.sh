python3 fid/fid_score.py 
--clean_dataset demogpairs \
--batch-size 128 \
--device cuda:2 \
--dims 2048 \
--save-stats \
--save-stats-dir ./debug/demogpairs \
--results-path ./debug/demogpairs/fid.json