# Grammar files

The grammar files defines a PCFG over the SCAN grammar. These can be used
to generate the data, using the `make_scan_data.py` script. The test
grammar creates the test data, while the remaining ones create the training
data for increasing support. The first number in the naming refers to the
number of verbs in the first embedded sentence (which is on one side for
the "and" conjunction and on the other for "after") while the second refers
to the number of verbs in second embedded sentence. In Experiment 2, we
increase the number of verbs in the support for one of the embedded
sentences, while we retain one verb from the other. In the test grammar,
this restricted verb is always present in the inverse embedded sentence.
