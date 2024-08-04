import boto3

# Initialize the Kinesis client
kinesis_client = boto3.client('kinesis', region_name='us-east-1')

# Get the shard iterator
response = kinesis_client.get_shard_iterator(
    StreamName='ride_predictions',
    ShardId='shardId-000000000000',
    ShardIteratorType='TRIM_HORIZON',
)

shard_iterator = response['ShardIterator']

# Read records from the stream
records_response = kinesis_client.get_records(ShardIterator=shard_iterator)

# Print out the records
for record in records_response['Records']:
    print(record['Data'].decode('utf-8'))
