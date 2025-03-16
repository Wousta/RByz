#!/bin/bash

# Run on server Node

srvr_ip=127.0.0.1
port=2000

build/clnt --srvr_ip $srvr_ip --port $port
