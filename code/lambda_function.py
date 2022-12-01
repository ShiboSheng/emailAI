import json
import boto3
import os
import logging
import email
from sagemaker.mxnet.model import MXNetPredictor
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

endpoint_name = os.environ['ENDPOINT_NAME']
vocabulary_length = 9013


def classify(msg):
    mxnet_pred = MXNetPredictor(endpoint_name)
    
    test_messages = [msg]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    result = mxnet_pred.predict(encoded_test_messages)
    print(result)
    return result


def reply(from_addr, to_addr, date, subject, message, label, confidence):
    length = min(240, len(message))
    message = message[:length]
    length = str(length)
    body = "We received your email sent at {} with the subject \"{}\".\n\nHere is a {} character sample of the email body:\n{}\n\nThe email was categorized as {} with a {}% confidence.".format(date, subject, length, message, label, confidence)
    client = boto3.client('ses')
    response = client.send_email(
        Destination={
            'ToAddresses': [
                from_addr,
            ],
        },
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': (body),
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': "Reply to your email",
            },
        },
        Source=to_addr,
    )
    print("Sent")
    

def parse(bucket_name, email_name):
    s3 = boto3.client('s3')
    obj_email = s3.get_object(Bucket=bucket_name, Key=email_name)
    raw_email = obj_email['Body'].read().decode('utf-8')
    parsed_email = email.message_from_string(raw_email)
    from_addr = email.utils.parseaddr(parsed_email['from'])[1]
    to_addr = parsed_email['to']
    date = email.utils.parsedate_to_datetime(parsed_email['date'])
    date = date.strftime("%m/%d/%Y, %H:%M:%S")
    subject = parsed_email['subject']
    body = parsed_email.get_payload()[0]
    original_msg = body.get_payload()
    msg = original_msg.replace('\r\n', ' ')
    msg = msg.replace('=', '')
    prediction = classify(msg)
    label = prediction['predicted_label'][0][0]
    confidence = prediction['predicted_probability'][0][0] * 100
    classification = 'SPAM'
    if label == 0.0:
        classification = 'NOT SPAM'
        confidence = 100 - confidence
    confidence = "{:.2f}".format(confidence)
    reply(from_addr, to_addr, date, subject, original_msg, classification, confidence)


def lambda_handler(event, context):
    for record in event['Records']:
        s3 = record['s3']
        bucket_name = s3['bucket']['name']
        email_name = s3['object']['key']
        logger.info("Received email: " + email_name)
        parse(bucket_name, email_name)

    return {
        'statusCode': 200,
        'body': json.dumps('Email parsed.')
    }