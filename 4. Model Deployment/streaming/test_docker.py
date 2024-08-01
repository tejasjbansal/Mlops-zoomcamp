import requests 

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49654476127272325694982812408478754550200875041920385026",
                "data": "eyJyaWRlIjogeyJQVUxvY2F0aW9uSUQiOiAxMzAsIkRPTG9jYXRpb25JRCI6IDIwNSwidHJpcF9kaXN0YW5jZSI6IDMuNjZ9LCAicmlkZV9pZCI6IDE1Nn0=",
                "approximateArrivalTimestamp": 1722533946.065
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49654476127272325694982812408478754550200875041920385026",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::975050357734:role/lambda-kinesis-role",
            "awsRegion": "us-east-1",
            "eventSourceARN": "arn:aws:kinesis:us-east-1:975050357734:stream/ride_events"
        }
    ]
}


url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
response = requests.post(url, json=event)
print(response.json())