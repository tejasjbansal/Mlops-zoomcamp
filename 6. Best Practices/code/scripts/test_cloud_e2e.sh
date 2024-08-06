export KINESIS_STREAM_INPUT="stg_ride_events-mlops-zoomcamp"
export KINESIS_STREAM_OUTPUT="stg_ride_predictions-mlops-zoomcamp"

# SHARD_ID=$(aws kinesis put-record \
# --stream-name ${KINESIS_STREAM_INPUT} \
# --partition-key 1 \
# --data file://scripts/data.b64 \
# --query 'ShardId' \
# --output text)

# sleep 60
export SHARD_ID="shardId-000000000001"
SHARD_ITERATOR=$(aws kinesis get-shard-iterator --shard-id ${SHARD_ID} --shard-iterator-type TRIM_HORIZON --stream-name ${KINESIS_STREAM_OUTPUT} --query 'ShardIterator')

aws kinesis get-records --shard-iterator $SHARD_ITERATOR