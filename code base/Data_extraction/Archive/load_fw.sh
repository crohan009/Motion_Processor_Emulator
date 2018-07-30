#!/bin/bash
JLinkGDBServer -Device LM4F232H5QD &
arm-none-eabi-gdb -x runme.gdb
pkill JLinkGDBServer