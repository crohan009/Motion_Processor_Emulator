target remote localhost:2331
monitor speed 1000
monitor flash device =  LM4F232H5QD
monitor halt
monitor reset
load ~/workspace_v7/LM4F_TM4C_uart/Release/LM4F_TM4C_uart.out
set confirm off
quit