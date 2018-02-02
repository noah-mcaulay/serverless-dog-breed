import os
import sys
import json
import boto3

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(HERE, "vendored"))

import tensorflow as tf


def return_lambda_gateway_response(code, body):
    return {"statusCode": code, "body": json.dumps(body)}

def predict(event, context):
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

    print("preparing to load file from s3....")

    s3 = boto3.resource('s3')
    bucket = s3.Bucket('serverless-dog-breed')

    print("finished loading file from s3!")

    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    labels = {}

    # change this as you see fit
    #image_path = sys.argv[1]

    # Read in the image_data
    image_data = tf.gfile.FastGFile("corgi.jpg", 'rb').read()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        labels = {}
        for node_id in top_k:
            labels[label_lines[node_id]] = predictions[0][node_id]

    return return_lambda_gateway_response(200, labels)
