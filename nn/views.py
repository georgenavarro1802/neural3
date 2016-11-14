#! /usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
from datetime import datetime

from django.db import transaction
from django.http import HttpResponseRedirect
from django.shortcuts import render_to_response
from numpy import array, exp, dot
from numpy import random

from neural3.settings import MEDIA_ROOT, BASE_DIR
from nn.forms import NeuralCSVForm
from nn.functions import ok_json, bad_json
from nn.models import NeuralFile, NeuralInputs


def conv(s):
    try:
        s = int(s)
    except ValueError:
        pass
    return s


def generate_file_name(nombre, original):
    ext = ""
    if original.find(".") > 0:
        ext = original[original.rfind("."):]
    fecha = datetime.now().date()
    return "{0}{1}{2}{3}".format(nombre, fecha.year, fecha.month, fecha.day, ext)


def datacsv(request):
    if request.method == 'POST':
        try:
            with transaction.atomic():
                # delete all neural file records and delete media files with django cleanup
                if NeuralFile.objects.exists():
                    NeuralFile.objects.all().delete()
                if NeuralInputs.objects.exists():
                    NeuralInputs.objects.all().delete()

                if 'csv' in request.FILES:
                    newfile = request.FILES['csv']
                    newfile._name = generate_file_name("csv_", newfile._name)

                    neuralfile = NeuralFile()
                    neuralfile.csv = newfile
                    neuralfile.save()

                    return ok_json()
                else:
                    return bad_json(mensaje="You must enter a valid csv file")
        except Exception:
            return bad_json(error=1)

    return render_to_response('datacsv.html', {'title': u"Input Data - CSV File (max 2Mb)", 'form': NeuralCSVForm()})


def index(request):
    csv_file = NeuralFile.objects.first() if NeuralFile.objects.exists() else None
    return render_to_response('base.html', {'title': u"Artificial Neural Network",
                                            'form': NeuralCSVForm(),
                                            'csv_file': csv_file})


def training(request):
    if request.method == 'POST':
        try:
            with transaction.atomic():

                # number of iterations from ajax request
                iterations = int(request.POST['iter'])

                # inputs from ajax
                n1 = int(request.POST['n1'])
                n2 = int(request.POST['n2'])
                n3 = int(request.POST['n3'])
                n4 = int(request.POST['n4'])
                n5 = int(request.POST['n5'])

                # csv file saved in model
                neural_file = NeuralFile.objects.first()

                # Read from csv file and fill list of inputs and ouputs
                # inputs are the first 5 columns and the last columns is the output
                inputs_l = []
                outputs_l = []
                output_folder = os.path.join(os.path.join(BASE_DIR, 'media'))
                with open(os.path.join(output_folder, neural_file.csv.name), 'rb') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for row in reader:
                        # Inputs list
                        inputs_l.append([conv(x) for x in row[:5]])
                        # Output list
                        outputs_l.append(conv(row[5]))

                # Input and Output convert list to numpy arrays
                inputs = array(inputs_l)
                outputs = array([outputs_l]).T  # .T is to transpose, y matrix has 5 rows with 1 column (vertical form)

                # unknown input
                new_input = array([n1, n2, n3, n4, n5])

                # initialize synapse_weights
                random.seed(1)
                # make use of machine learning formula Delta rule.
                weights = 2 * random.random((5, 1)) - 1

                # output without untrained neuron usign Sigmoid Function
                # which always return a value between -1 or 1: f(x) = 1 / 1 + e exp (-(input * weight))
                # print 1 / (1 + exp(-(dot(new_input, weights))))

                # Train the network in a range of n iterations
                weights_average = 0
                for i in xrange(iterations):
                    # Calculate the value for the each of the examples
                    output = 1 / (1 + exp(-(dot(inputs, weights))))
                    # Run the adjustments to weights
                    weights += dot(inputs.T, (outputs - output) * (output * (1 - output)))
                    # print "Output: %s" % output
                    # print "Weights: %s" % weights

                weights_average = dot(sum(weights)/5.0, 100.0)
                # print weights_average

                value_with_train = 1 / (1 + exp(-(dot(new_input, weights))))
                if value_with_train > 0.5:
                    possible_result = 1
                elif value_with_train < 0.5:
                    possible_result = 0
                else:
                    possible_result = 0.5

                return ok_json(data={'possible_result': possible_result,
                                     'value_with_train': round(value_with_train[0], 6),
                                     'weights_average': round(100 - weights_average[0], 2)})

        except Exception as ex:
            return bad_json(error=1)

    return bad_json(error=0)

