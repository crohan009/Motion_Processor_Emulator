//*****************************************************************************
//
// uart_echo.c - Example for reading data from and writing data to the UART in
//               an interrupt driven fashion.
//
// Copyright (c) 2011-2012 Texas Instruments Incorporated.  All rights reserved.
// Software License Agreement
//
// Texas Instruments (TI) is supplying this software for use solely and
// exclusively on TI's microcontroller products. The software is owned by
// TI and/or its suppliers, and is protected under applicable copyright
// laws. You may not combine this software with "viral" open-source
// software in order to form a larger program.
//
// THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
// NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
// NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
// CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
// DAMAGES, FOR ANY REASON WHATSOEVER.
//
// This is part of revision 9453 of the EK-LM4F232 Firmware Package.
//
//*****************************************************************************
# define PI           3.14159265358979323846  /* pi */

#include <stdio.h>
#include <errno.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_i2c.h"
#include "driverlib/debug.h"
#include "driverlib/fpu.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"
#include "driverlib/rom.h"
#include "driverlib/systick.h"
#include "driverlib/pin_map.h"
#include "driverlib/uart.h"
#include "driverlib/i2c.h"
#include "include/uartstdio.h"
//#include "grlib/grlib.h"
//#include "drivers/cfal96x64x16.h"


//#include "LM4F_TM4C_libs.h"
//#include "systick.h"
#include "MPU_interface/mpu_interface.h"
#include "i2c_functions/i2c_TM4C.h"
#include "include/imu.h"


int *f;
uint8_t *data = 0;
imu_scaled_data b;
int result;
long gyro[3], accel[3];

typedef unsigned char uint8;

#define USE_PLL_CLOCK

#define AP_UART UART1_BASE

// send n bytes to the RM48
//void AP_send(int n, char *s)
//{
//	int i;
//	for (i=0; i<n; i++)
//	{
//		ROM_UARTCharPut(AP_UART,s[i]);
//	}
//}

// some simple delay functions
double delay()
{
	double _s = 0;
	int k;
	for (k=0; k<50000; k++) _s += 1./k;
	return _s;
}
double delay2(int k1)
{
	double _s = 0;
	int k;
	for (k=0; k<500*k1; k++) _s += 1./k;
	return _s;
}

// send 1 byte to AP_UART
void AP_send_char(char s)
{
    ROM_UARTCharPut(AP_UART,s);
}

// send n bytes to AP_UART
void AP_send(int n, const char *s)
{
    int i;
    for (i=0; i<n; i++)
    {
        ROM_UARTCharPut(AP_UART,s[i]);
    }
}

// send a string message to AP_UART
void send_msg(const char *s)
{
  send(strlen(s), s);
}

void send_char(char s)
{
    ROM_UARTCharPut(UART0_BASE,s);
}

// send n bytes to AP_UART
void send(int n, const char *s)
{
    int i;
    for (i=0; i<n; i++)
    {
        ROM_UARTCharPut(UART0_BASE,s[i]);
    }
}

// send a string message to AP_UART
void AP_send_msg(const char *s)
{
  AP_send(strlen(s), s);
}

// system tick (systick)

//static volatile int systick_n = 0;
volatile uint32_t systick_n= 0;

void systick_int_handler()
{
	systick_n ++;
}

void enable_systick()
{
	send_msg("enabling systick");
	SysTickPeriodSet(160000);
	SysTickIntRegister(&systick_int_handler);
	SysTickIntEnable();
	SysTickEnable();
}

// Delay for a specified number of milliseconds
//void delay_ms(unsigned long num_ms){
//	int systick_n1 = systick_n;
//	while (systick_n < (systick_n1 + num_ms));
//}

void delay_ms (unsigned  long time  )
{
    SysCtlDelay(time * (SysCtlClockGet() / 3 / 1000));
}


// Returns the current time in milliseconds
long get_ms()
{
	return systick_n;
}

// Pauses until the time (in milliseconds) given in the argument
void wait_until(long t)
{
	while (systick_n < t);
}

void print_float(double v)
{
  int decimal = 2;
  int i = 1;
  int intPart, fractPart;
  for (;decimal!=0; i*=10, decimal--);
  intPart = (int)v;
  fractPart = (int)((v-(double)(int)v)*i);
  if(fractPart < 0) fractPart *= -1;
  UARTprintf("%i.%i", intPart, fractPart);
}


////*****************************************************************************
////
//// The UART interrupt handlers.
////
////*****************************************************************************
//
//void
//UARTIntHandler(void)
//{
//    uint32_t ui32Status;
//    ui32Status = UARTIntStatus(UART0_BASE, true);   // Get the interrrupt status
//    UARTIntClear(UART0_BASE, ui32Status);    // Clear the asserted interrupts.
//}
//
//
void ConfigureUART(void)
{
    //UART0
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);// Enable the GPIO Peripheral used by the UART.
    SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0); // Enable UART0

    // Configure GPIO Pins for UART mode.
    GPIOPinConfigure(GPIO_PA0_U0RX);
    GPIOPinConfigure(GPIO_PA1_U0TX);
    GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);    // Use the internal 16MHz oscillator as the UART clock source.
    UARTStdioConfig(0, 115200, 16000000);    // Initialize the UART for console I/O.
}

void test_invensense()
{
    unsigned char _whoami;
    i2c_common_setup();
    i2c_read(104,117,1,&_whoami);
    mpu_open(200,1);
    EulerAngles_outputs angs;
    MPU_MPL_outputs mpu_mpl_outputs;
    RotationMatrix_outputs R;
    while (1)
    {
        long _t = get_ms();
        int data_updated = mpu_update(&mpu_mpl_outputs);
        mpu_get_euler(&angs); // angs.angles[3] now has Euler angles (in radians)
        char s[64];
        sprintf(s,"Euler PRY : %.2f %.2f %.2f\r\n", data_updated, angs.angles[0]*180.0/3.1415926, angs.angles[1]*180.0/3.1415926, angs.angles[2]*180.0/3.1415926);
        send_msg(s);
        mpu_get_rot_mat(&R);  // R.R[9] now has rotation matrix elements
        wait_until(_t + 5);   // wait for 5 milliseconds
    }
}

//*****************************************************************************
//
// The UART interrupt handlers.
//
//*****************************************************************************
void
UART0IntHandler(void)
{
    unsigned long ulStatus;

    //
    // Get the interrupt status.
    //
    ulStatus = ROM_UARTIntStatus(UART0_BASE, true);

    //
    // Clear the asserted interrupts.
    //
    ROM_UARTIntClear(UART0_BASE, ulStatus);

    //
    // Loop while there are characters in the receive FIFO.
    //
    while(ROM_UARTCharsAvail(UART0_BASE))
    {
        //
        // Read the next character from the UART and write it back to the UART.
        //
        unsigned char c = ROM_UARTCharGetNonBlocking(UART0_BASE);
        //ROM_UARTCharPutNonBlocking(UART2_BASE, c);
    }
}


int main(void)
{
  // initialize communications structures
      //init_AP_comms();
   
    //
    // Enable lazy stacking for interrupt handlers.  This allows floating-point
    // instructions to be used within interrupt handlers, but at the expense of
    // extra stack usage.
    //
    ROM_FPULazyStackingEnable();

#ifdef USE_PLL_CLOCK
    ROM_SysCtlClockSet(SYSCTL_SYSDIV_2_5 | SYSCTL_USE_PLL | SYSCTL_OSC_MAIN | SYSCTL_XTAL_16MHZ);
#else
    ROM_SysCtlClockSet(SYSCTL_SYSDIV_1 | SYSCTL_USE_OSC | SYSCTL_OSC_MAIN | SYSCTL_XTAL_16MHZ);
#endif

    ConfigureUART();
    enable_systick();

    i2c_common_setup();
    UARTprintf("######################## MPU-9150 ########################\n");
    i2c_read(104, 117, 1, &data);
    UARTprintf("Who am I : %04x\r\n",&data[0]);
    mpu_open(200,1);
   //    result = mpu_run_self_test(gyro, accel);
   //    UARTprintf("Test Result : %d\r\n", result);
    EulerAngles_outputs angs;
    MPU_MPL_outputs mpu_mpl_outputs;
    RotationMatrix_outputs R;
    UARTprintf("MPU initialized \n");
    delay_ms(1000);

//        UARTprintf("######################## LSM9DS0 ########################\n");
//        UARTprintf("Who_am_I: %04x\n",I2CReceive(ADDR_ACCEL_LSM9,0x0f));
//        delay_ms(1000);
//        init_imu_scaledStruct(&b, SCALE_2G, SCALE_2GAUSS, SCALE_250_DPS, UNIT_M_SQUARED);
//        init_imu(&b);
//        UARTprintf("LSM initialized \n");
//        delay_ms(1000);
//

   test_invensense();
   while (1){
//     long _t = get_ms();
//                 int data_updated = mpu_update(&mpu_mpl_outputs);
//       //          read_scaled_imu_data(&b);
//       //  //        b.acc1_data[0], b.acc1_data[1], b.acc1_data[2], b.gyro_data[0], b.gyro_data[1], b.gyro_data[2],  b.mag_data[0],  b.mag_data[1], b.mag_data[2]
//       //  //      UARTprintf("Update: %d\r\n",data_updated);
//                 mpu_get_euler(&angs); // angs.angles[3] now has Euler angles (in radians)
//                 char s[64];
//                 sprintf(s,"Euler PRY : %0.2f;%0.2f;%0.2f\r\n",angs.angles[0]* 180/PI,angs.angles[1]* 180/PI,angs.angles[2]* 180/PI);
//                 send_msg(s);
   }
   // to send positioning_update message:
   // positioning_update_t positioning_update_msg;
   // AP_send_positioning_update_msg(&positioning_update_msg);
}
